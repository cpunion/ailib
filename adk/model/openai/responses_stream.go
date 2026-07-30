package openai

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

// responsesAccumulator consumes an OpenAI Responses API SSE event stream
// (https://platform.openai.com/docs/api-reference/responses-streaming) into the
// pieces of a single model.LLMResponse. It is shared by the generic
// /v1/responses model and the Codex backend variant, which differ only in
// request construction and transport headers — the streamed event shapes are
// identical.
//
// handleEvent is intentionally yield-free: it accumulates state and returns the
// incremental output text (for callers that stream partials) plus a fatal error
// for terminal failure events, leaving the iter.Seq2 plumbing to the caller.
type responsesAccumulator struct {
	textBuilder   strings.Builder
	calls         []*codexResponseCall
	callsByID     map[string]*codexResponseCall
	outputs       []*responsesOrderedOutput
	outputsByID   map[string]*responsesOrderedOutput
	usage         *genai.GenerateContentResponseUsageMetadata
	finish        genai.FinishReason
	terminal      bool
	responseID    string
	responseModel string
}

type codexResponseReasoning struct {
	rawJSON json.RawMessage
	summary string
}

type responsesOrderedOutput struct {
	id        string
	kind      string
	reasoning *codexResponseReasoning
	call      *codexResponseCall
	image     *genai.Blob
	text      strings.Builder
	done      bool
}

func newResponsesAccumulator() *responsesAccumulator {
	return &responsesAccumulator{
		callsByID:   map[string]*codexResponseCall{},
		outputsByID: map[string]*responsesOrderedOutput{},
		finish:      genai.FinishReasonStop,
	}
}

func (a *responsesAccumulator) ensureOutput(
	itemID string,
	kind string,
) *responsesOrderedOutput {
	itemID = strings.TrimSpace(itemID)
	if itemID != "" {
		if existing := a.outputsByID[itemID]; existing != nil {
			if existing.kind == "" {
				existing.kind = kind
			}
			return existing
		}
	}
	output := &responsesOrderedOutput{id: itemID, kind: kind}
	a.outputs = append(a.outputs, output)
	if itemID != "" {
		a.outputsByID[itemID] = output
	}
	return output
}

func (a *responsesAccumulator) ensureCall(itemID string) *codexResponseCall {
	if itemID == "" {
		itemID = "call_" + uuid.NewString()[:8]
	}
	if existing := a.callsByID[itemID]; existing != nil {
		return existing
	}
	call := &codexResponseCall{itemID: itemID}
	a.callsByID[itemID] = call
	a.calls = append(a.calls, call)
	return call
}

// handleEvent folds one parsed SSE event into the accumulator. deltaText is the
// newly produced output text (non-empty only for output_text deltas); the
// caller emits a partial response when it is non-empty and streaming is on.
// A non-nil error is terminal: the stream carried an explicit failure.
func (a *responsesAccumulator) handleEvent(event codexResponsesEvent) (deltaText string, err error) {
	if event.Response != nil {
		if strings.TrimSpace(event.Response.ID) != "" {
			a.responseID = strings.TrimSpace(event.Response.ID)
		}
		if strings.TrimSpace(event.Response.Model) != "" {
			a.responseModel = strings.TrimSpace(event.Response.Model)
		}
	}
	switch event.Type {
	case "response.output_text.delta":
		if event.Delta == "" {
			return "", nil
		}
		a.textBuilder.WriteString(event.Delta)
		if strings.TrimSpace(event.ItemID) != "" {
			output := a.ensureOutput(event.ItemID, "message")
			output.text.WriteString(event.Delta)
		}
		return event.Delta, nil
	case "response.function_call_arguments.delta":
		call := a.ensureCall(event.ItemID)
		call.arguments.WriteString(event.Delta)
	case "response.output_item.added", "response.output_item.done":
		if event.Item == nil {
			return "", nil
		}
		itemID := firstNonEmptyString(
			event.Item.ID,
			event.Item.CallID,
			event.ItemID,
		)
		output := a.ensureOutput(itemID, event.Item.Type)
		switch event.Item.Type {
		case "reasoning":
			if event.Type != "response.output_item.done" ||
				len(event.Item.rawJSON) == 0 {
				return "", nil
			}
			if output.done {
				return "", nil
			}
			output.reasoning = &codexResponseReasoning{
				rawJSON: append(json.RawMessage(nil), event.Item.rawJSON...),
				summary: responsesReasoningSummary(*event.Item),
			}
			output.done = true
		case "function_call":
			call := a.ensureCall(itemID)
			output.call = call
			if strings.TrimSpace(event.Item.CallID) != "" {
				call.callID = strings.TrimSpace(event.Item.CallID)
			}
			if strings.TrimSpace(event.Item.Name) != "" {
				call.name = strings.TrimSpace(event.Item.Name)
			}
			if strings.TrimSpace(event.Item.Arguments) != "" {
				call.arguments.Reset()
				call.arguments.WriteString(strings.TrimSpace(event.Item.Arguments))
			}
			if event.Type == "response.output_item.done" {
				output.done = true
			}
		case "message":
			if event.Type != "response.output_item.done" {
				return "", nil
			}
			var completedText strings.Builder
			for _, part := range event.Item.Content {
				if strings.EqualFold(
					strings.TrimSpace(part.Type),
					"output_text",
				) && part.Text != "" {
					completedText.WriteString(part.Text)
				}
			}
			if completedText.Len() > 0 {
				output.text.Reset()
				output.text.WriteString(completedText.String())
				if a.textBuilder.Len() == 0 {
					a.textBuilder.WriteString(completedText.String())
				}
			}
			output.done = true
		case "image_generation_call":
			if event.Type != "response.output_item.done" || strings.TrimSpace(event.Item.Result) == "" {
				return "", nil
			}
			if output.done {
				return "", nil
			}
			imageData, decodeErr := base64.StdEncoding.DecodeString(strings.TrimSpace(event.Item.Result))
			if decodeErr != nil {
				return "", fmt.Errorf("failed to decode responses image output: %w", decodeErr)
			}
			output.image = &genai.Blob{
				MIMEType: detectCodexImageMimeType(imageData),
				Data:     imageData,
			}
			output.done = true
		}
	case "response.completed", "response.incomplete":
		if event.Response == nil {
			return "", fmt.Errorf(
				"responses terminal event is missing response",
			)
		}
		a.mergeTerminalReasoning(event.Response.Output)
		a.usage = buildCodexUsageMetadata(event.Response.Usage)
		status := strings.ToLower(strings.TrimSpace(event.Response.Status))
		if event.Type == "response.incomplete" {
			status = "incomplete"
		}
		switch status {
		case "completed", "":
			a.finish = genai.FinishReasonStop
		case "incomplete":
			if event.Response.IncompleteDetails != nil && strings.EqualFold(strings.TrimSpace(event.Response.IncompleteDetails.Reason), "max_output_tokens") {
				a.finish = genai.FinishReasonMaxTokens
			} else {
				a.finish = genai.FinishReasonOther
			}
		default:
			message := "responses request failed"
			if event.Response.Error != nil && strings.TrimSpace(event.Response.Error.Message) != "" {
				message = strings.TrimSpace(event.Response.Error.Message)
			}
			return "", fmt.Errorf("%s", message)
		}
		a.terminal = true
	case "response.failed":
		message := codexStreamErrorMessage(event)
		if message == "" {
			message = "responses request failed"
		}
		return "", fmt.Errorf("%s", message)
	case "error":
		if msg := codexStreamErrorMessage(event); msg != "" {
			return "", fmt.Errorf("%s", msg)
		}
		return "", fmt.Errorf("responses stream error")
	}
	return "", nil
}

func (a *responsesAccumulator) mergeTerminalReasoning(
	items []codexResponsesOutputItem,
) {
	for index := range items {
		item := &items[index]
		if item.Type != "reasoning" || len(item.rawJSON) == 0 {
			continue
		}
		id := strings.TrimSpace(item.ID)
		if existing := a.outputsByID[id]; id != "" &&
			existing != nil &&
			existing.reasoning != nil {
			existing.reasoning.rawJSON =
				backfillReasoningEncryptedContentJSON(
					existing.reasoning.rawJSON,
					item.rawJSON,
				)
			continue
		}
		output := a.ensureOutput(id, "reasoning")
		output.reasoning = &codexResponseReasoning{
			rawJSON: append(json.RawMessage(nil), item.rawJSON...),
			summary: responsesReasoningSummary(*item),
		}
		output.done = true
	}
}

func responsesReasoningSummary(item codexResponsesOutputItem) string {
	var summary []string
	for _, part := range item.Summary {
		if strings.EqualFold(
			strings.TrimSpace(part.Type),
			"summary_text",
		) && strings.TrimSpace(part.Text) != "" {
			summary = append(summary, strings.TrimSpace(part.Text))
		}
	}
	return strings.Join(summary, "\n")
}

func backfillReasoningEncryptedContentJSON(
	completed json.RawMessage,
	terminal json.RawMessage,
) json.RawMessage {
	original := append(json.RawMessage(nil), completed...)
	var completedFields map[string]json.RawMessage
	if err := json.Unmarshal(completed, &completedFields); err != nil {
		return original
	}
	var current string
	if raw := completedFields["encrypted_content"]; len(raw) > 0 {
		_ = json.Unmarshal(raw, &current)
	}
	if strings.TrimSpace(current) != "" {
		return original
	}

	var terminalFields map[string]json.RawMessage
	if err := json.Unmarshal(terminal, &terminalFields); err != nil {
		return original
	}
	encryptedRaw := terminalFields["encrypted_content"]
	var encrypted string
	if len(encryptedRaw) == 0 ||
		json.Unmarshal(encryptedRaw, &encrypted) != nil ||
		strings.TrimSpace(encrypted) == "" {
		return original
	}
	completedFields["encrypted_content"] =
		append(json.RawMessage(nil), encryptedRaw...)
	merged, err := json.Marshal(completedFields)
	if err != nil {
		return original
	}
	return merged
}

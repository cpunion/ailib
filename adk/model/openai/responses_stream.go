package openai

import (
	"encoding/base64"
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
	textBuilder strings.Builder
	calls       []*codexResponseCall
	callsByID   map[string]*codexResponseCall
	images      []*genai.Blob
	seenImages  map[string]struct{}
	usage       *genai.GenerateContentResponseUsageMetadata
	finish      genai.FinishReason
}

func newResponsesAccumulator() *responsesAccumulator {
	return &responsesAccumulator{
		callsByID:  map[string]*codexResponseCall{},
		seenImages: map[string]struct{}{},
		finish:     genai.FinishReasonStop,
	}
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
	switch event.Type {
	case "response.output_text.delta":
		if event.Delta == "" {
			return "", nil
		}
		a.textBuilder.WriteString(event.Delta)
		return event.Delta, nil
	case "response.function_call_arguments.delta":
		call := a.ensureCall(event.ItemID)
		call.arguments.WriteString(event.Delta)
	case "response.output_item.added", "response.output_item.done":
		if event.Item == nil {
			return "", nil
		}
		switch event.Item.Type {
		case "function_call":
			call := a.ensureCall(firstNonEmptyString(event.Item.ID, event.Item.CallID, event.ItemID))
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
		case "message":
			if a.textBuilder.Len() == 0 {
				for _, part := range event.Item.Content {
					if strings.EqualFold(strings.TrimSpace(part.Type), "output_text") && part.Text != "" {
						a.textBuilder.WriteString(part.Text)
					}
				}
			}
		case "image_generation_call":
			itemID := firstNonEmptyString(event.Item.ID, event.Item.CallID, event.ItemID)
			if event.Type != "response.output_item.done" || strings.TrimSpace(event.Item.Result) == "" {
				return "", nil
			}
			if _, exists := a.seenImages[itemID]; exists {
				return "", nil
			}
			imageData, decodeErr := base64.StdEncoding.DecodeString(strings.TrimSpace(event.Item.Result))
			if decodeErr != nil {
				return "", fmt.Errorf("failed to decode responses image output: %w", decodeErr)
			}
			a.images = append(a.images, &genai.Blob{
				MIMEType: detectCodexImageMimeType(imageData),
				Data:     imageData,
			})
			a.seenImages[itemID] = struct{}{}
		}
	case "response.completed":
		if event.Response == nil {
			return "", nil
		}
		a.usage = buildCodexUsageMetadata(event.Response.Usage)
		switch strings.ToLower(strings.TrimSpace(event.Response.Status)) {
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
	case "error":
		if msg := codexStreamErrorMessage(event); msg != "" {
			return "", fmt.Errorf("%s", msg)
		}
		return "", fmt.Errorf("responses stream error")
	}
	return "", nil
}

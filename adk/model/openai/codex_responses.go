package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
	"time"

	providercontract "github.com/cpunion/ailib/adk/model/provider"
	"github.com/google/uuid"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

const codexSessionIDMaxRunes = 64

type codexResponsesModel struct {
	modelName  string
	config     *ClientConfig
	accountID  string
	httpClient *http.Client
}

type codexResponsesRequest struct {
	Model             string                   `json:"model"`
	Store             bool                     `json:"store"`
	Stream            bool                     `json:"stream"`
	Instructions      string                   `json:"instructions,omitempty"`
	Input             []any                    `json:"input,omitempty"`
	Tools             []codexResponsesTool     `json:"tools,omitempty"`
	ToolChoice        any                      `json:"tool_choice,omitempty"`
	ParallelToolCalls bool                     `json:"parallel_tool_calls"`
	Reasoning         *codexResponsesReasoning `json:"reasoning,omitempty"`
	Text              *codexResponsesText      `json:"text,omitempty"`
	Include           []string                 `json:"include,omitempty"`
	PromptCacheKey    string                   `json:"prompt_cache_key,omitempty"`
}

type codexResponsesReasoning struct {
	Effort  string `json:"effort"`
	Summary string `json:"summary,omitempty"`
}

const responsesReasoningSignatureVersion = "ailib.responses.reasoning.v1"

type responsesReasoningSignature struct {
	Version       string          `json:"version"`
	Provider      string          `json:"provider"`
	Model         string          `json:"model"`
	ResponseModel string          `json:"responseModel,omitempty"`
	Item          json.RawMessage `json:"item"`
}

type codexResponsesText struct {
	Verbosity string `json:"verbosity"`
}

type codexResponsesTool struct {
	Type              string         `json:"type"`
	Name              string         `json:"name,omitempty"`
	Description       string         `json:"description,omitempty"`
	Parameters        map[string]any `json:"parameters,omitempty"`
	Size              string         `json:"size,omitempty"`
	Quality           string         `json:"quality,omitempty"`
	OutputFormat      string         `json:"output_format,omitempty"`
	OutputCompression *int           `json:"output_compression,omitempty"`
	Background        string         `json:"background,omitempty"`
	Action            string         `json:"action,omitempty"`
}

type codexResponsesEvent struct {
	Type     string                    `json:"type"`
	Delta    string                    `json:"delta,omitempty"`
	Message  string                    `json:"message,omitempty"`
	Code     string                    `json:"code,omitempty"`
	Error    *codexResponsesError      `json:"error,omitempty"`
	ItemID   string                    `json:"item_id,omitempty"`
	Item     *codexResponsesOutputItem `json:"item,omitempty"`
	Response *codexResponsesFinal      `json:"response,omitempty"`
}

type codexResponsesOutputItem struct {
	ID            string                        `json:"id,omitempty"`
	Type          string                        `json:"type"`
	CallID        string                        `json:"call_id,omitempty"`
	Name          string                        `json:"name,omitempty"`
	Arguments     string                        `json:"arguments,omitempty"`
	Content       []codexResponsesOutputContent `json:"content,omitempty"`
	Status        string                        `json:"status,omitempty"`
	RevisedPrompt string                        `json:"revised_prompt,omitempty"`
	Result        string                        `json:"result,omitempty"`
	Summary       []codexResponsesOutputContent `json:"summary,omitempty"`
	rawJSON       json.RawMessage
}

func (i *codexResponsesOutputItem) UnmarshalJSON(data []byte) error {
	type itemAlias codexResponsesOutputItem
	var decoded itemAlias
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}
	*i = codexResponsesOutputItem(decoded)
	i.rawJSON = append(i.rawJSON[:0], data...)
	return nil
}

type codexResponsesOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type codexResponsesFinal struct {
	ID                string                     `json:"id,omitempty"`
	Model             string                     `json:"model,omitempty"`
	Status            string                     `json:"status"`
	Output            []codexResponsesOutputItem `json:"output,omitempty"`
	Usage             *codexResponsesUsage       `json:"usage,omitempty"`
	Error             *codexResponsesError       `json:"error,omitempty"`
	IncompleteDetails *codexIncompleteReason     `json:"incomplete_details,omitempty"`
}

type codexResponsesUsage struct {
	InputTokens        int `json:"input_tokens"`
	OutputTokens       int `json:"output_tokens"`
	TotalTokens        int `json:"total_tokens"`
	InputTokensDetails *struct {
		CachedTokens int `json:"cached_tokens,omitempty"`
	} `json:"input_tokens_details,omitempty"`
}

type codexResponsesError struct {
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
	Type    string `json:"type,omitempty"`
}

type codexIncompleteReason struct {
	Reason string `json:"reason"`
}

type codexResponseCall struct {
	itemID    string
	callID    string
	name      string
	arguments strings.Builder
}

func NewCodexResponsesModel(ctx context.Context, modelName string, config *ClientConfig, accountID string) (model.LLM, error) {
	_ = ctx
	if config == nil {
		config = &ClientConfig{}
	}
	if strings.TrimSpace(config.APIKey) == "" {
		return nil, fmt.Errorf("codex responses: missing API key")
	}
	if strings.TrimSpace(config.BaseURL) == "" {
		return nil, fmt.Errorf("codex responses: missing base URL")
	}
	if strings.TrimSpace(accountID) == "" {
		return nil, fmt.Errorf("codex responses: missing chatgpt account id")
	}
	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &codexResponsesModel{
		modelName:  modelName,
		config:     config,
		accountID:  accountID,
		httpClient: httpClient,
	}, nil
}

func (m *codexResponsesModel) Name() string {
	return m.modelName
}

func (m *codexResponsesModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	(&openAIModel{}).maybeAppendUserContent(req)
	codexReq, err := m.convertRequest(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, fmt.Errorf("failed to convert codex request: %w", err))
		}
	}
	return func(yield func(*model.LLMResponse, error) bool) {
		start := time.Now()
		attempt := m.newAttempt()
		httpResp, err := m.sendRequest(ctx, codexReq)
		if err != nil {
			m.applyErrorAttempt(&attempt, err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		defer httpResp.Body.Close()
		attempt.StatusCode = httpResp.StatusCode

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4<<20)

		acc := newResponsesAccumulator()
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimSpace(strings.TrimPrefix(line, "data: "))
			if data == "" || data == "[DONE]" {
				continue
			}

			var event codexResponsesEvent
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				err = fmt.Errorf(
					"decode codex responses stream event: %w",
					err,
				)
				m.applyErrorAttempt(&attempt, err)
				m.observeAttempt(attempt, start)
				yield(nil, err)
				return
			}

			delta, evErr := acc.handleEvent(event)
			if evErr != nil {
				m.applyErrorAttempt(&attempt, evErr)
				m.observeAttempt(attempt, start)
				yield(nil, evErr)
				return
			}
			if delta != "" && stream {
				if !yield(&model.LLMResponse{
					Content: &genai.Content{
						Role:  "model",
						Parts: []*genai.Part{{Text: delta}},
					},
					Partial: true,
				}, nil) {
					return
				}
			}
		}
		if err := scanner.Err(); err != nil {
			err = fmt.Errorf("codex responses stream error: %w", err)
			m.applyErrorAttempt(&attempt, err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		if !acc.terminal {
			err := fmt.Errorf(
				"codex responses stream ended without a terminal event",
			)
			m.applyErrorAttempt(&attempt, err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}

		finalResp, err := buildCodexFinalResponse(
			acc,
			"codex",
			m.modelName,
			acc.usage,
			acc.finish,
		)
		if err != nil {
			m.applyErrorAttempt(&attempt, err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		applyCodexSuccessAttempt(
			&attempt, acc.usage, acc.finish, m.accountID, acc.responseID,
		)
		m.observeAttempt(attempt, start)
		yield(finalResp, nil)
	}
}

func codexStreamErrorMessage(event codexResponsesEvent) string {
	var parts []string
	if msg := strings.TrimSpace(event.Message); msg != "" {
		parts = append(parts, msg)
	}
	if code := strings.TrimSpace(event.Code); code != "" {
		parts = append(parts, "code="+code)
	}
	appendErr := func(err *codexResponsesError) {
		if err == nil {
			return
		}
		if msg := strings.TrimSpace(err.Message); msg != "" {
			parts = append(parts, msg)
		}
		if typ := strings.TrimSpace(err.Type); typ != "" {
			parts = append(parts, "type="+typ)
		}
		if code := strings.TrimSpace(err.Code); code != "" {
			parts = append(parts, "code="+code)
		}
	}
	appendErr(event.Error)
	if event.Response != nil {
		appendErr(event.Response.Error)
	}
	if len(parts) == 0 {
		return ""
	}
	return strings.Join(dedupeStrings(parts), "; ")
}

func dedupeStrings(in []string) []string {
	seen := map[string]struct{}{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func (m *codexResponsesModel) convertRequest(req *model.LLMRequest) (*codexResponsesRequest, error) {
	out := &codexResponsesRequest{
		Model:             m.modelName,
		Store:             false,
		Stream:            true,
		ParallelToolCalls: true,
		Include:           []string{"reasoning.encrypted_content"},
		PromptCacheKey:    clampCodexSessionID(m.config.PromptCacheKey),
	}

	if req.Config != nil && req.Config.SystemInstruction != nil {
		out.Instructions = extractTextFromContent(req.Config.SystemInstruction)
	}
	if strings.TrimSpace(out.Instructions) == "" {
		out.Instructions = defaultCodexInstructions()
	}
	effort := strings.TrimSpace(m.config.ReasoningEffort)
	if effort == "" {
		effort = resolveReasoningEffort("codex", out.Instructions)
	}
	if effort != "" {
		normalized, err := normalizeCodexReasoningEffort(effort)
		if err != nil {
			return nil, err
		}
		summary, err := normalizeCodexReasoningSummary(m.config.ReasoningSummary)
		if err != nil {
			return nil, err
		}
		out.Reasoning = &codexResponsesReasoning{
			Effort:  normalized,
			Summary: summary,
		}
	} else if strings.TrimSpace(m.config.ReasoningSummary) != "" {
		return nil, fmt.Errorf("codex responses: reasoning summary requires reasoning effort")
	}
	verbosity, err := normalizeCodexTextVerbosity(m.config.TextVerbosity)
	if err != nil {
		return nil, err
	}
	out.Text = &codexResponsesText{Verbosity: verbosity}

	for _, content := range req.Contents {
		items, err := convertResponsesInputContent(
			content,
			"codex",
			m.modelName,
		)
		if err != nil {
			return nil, err
		}
		out.Input = append(out.Input, items...)
	}

	if req.Config != nil {
		for _, tool := range req.Config.Tools {
			if tool.FunctionDeclarations == nil {
				continue
			}
			for _, fn := range tool.FunctionDeclarations {
				out.Tools = append(out.Tools, codexResponsesTool{
					Type:        "function",
					Name:        fn.Name,
					Description: fn.Description,
					Parameters:  stripCodexToolMetaSchema(convertFunctionParameters(fn)),
				})
			}
		}
		if imageTool, ok := buildCodexImageTool(req.Config); ok {
			out.Tools = append(out.Tools, imageTool)
		}
	}
	switch {
	case hasOnlyCodexImageTool(out.Tools):
		out.ToolChoice = map[string]any{"type": "image_generation"}
	case len(out.Tools) > 0:
		out.ToolChoice = "auto"
	}
	return out, nil
}

func normalizeCodexReasoningEffort(raw string) (string, error) {
	value := strings.ToLower(strings.TrimSpace(raw))
	switch value {
	case "extra-high", "extra_high", "extrahigh", "x-high", "x_high":
		value = "xhigh"
	}
	switch value {
	case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		return value, nil
	default:
		return "", fmt.Errorf("codex responses: unsupported reasoning effort %q", raw)
	}
}

func normalizeCodexReasoningSummary(raw string) (string, error) {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" {
		return "auto", nil
	}
	switch value {
	case "auto", "concise", "detailed", "off", "on":
		return value, nil
	default:
		return "", fmt.Errorf("codex responses: unsupported reasoning summary %q", raw)
	}
}

func normalizeCodexTextVerbosity(raw string) (string, error) {
	value := strings.ToLower(strings.TrimSpace(raw))
	if value == "" {
		return "low", nil
	}
	switch value {
	case "low", "medium", "high":
		return value, nil
	default:
		return "", fmt.Errorf("codex responses: unsupported text verbosity %q", raw)
	}
}

func clampCodexSessionID(raw string) string {
	value := strings.TrimSpace(raw)
	runes := []rune(value)
	if len(runes) <= codexSessionIDMaxRunes {
		return value
	}
	return string(runes[:codexSessionIDMaxRunes])
}

// convertResponsesInputContent maps a genai.Content into OpenAI Responses API
// input items (message / function_call / function_call_output). It is shared by
// the Codex backend and the generic /v1/responses model.
func convertResponsesInputContent(
	content *genai.Content,
	provider string,
	modelName string,
) ([]any, error) {
	if content == nil || len(content.Parts) == 0 {
		return nil, nil
	}

	role := strings.ToLower(strings.TrimSpace(content.Role))
	if role == "" {
		role = "user"
	}
	if role == "model" {
		role = "assistant"
	}

	messageParts := make([]map[string]any, 0, len(content.Parts))
	items := make([]any, 0, len(content.Parts))
	flushMessage := func() {
		if len(messageParts) == 0 {
			return
		}
		items = append(items, map[string]any{
			"type":    "message",
			"role":    role,
			"content": messageParts,
		})
		messageParts = nil
	}

	for _, part := range content.Parts {
		switch {
		case part == nil:
			continue
		case part.Thought && len(part.ThoughtSignature) > 0:
			raw, ok := decodeResponsesReasoningSignature(
				part.ThoughtSignature,
				provider,
				modelName,
			)
			if ok {
				flushMessage()
				items = append(items, raw)
				continue
			}
			if part.Text != "" {
				messageParts = append(messageParts, map[string]any{
					"type": "output_text",
					"text": part.Text,
				})
			}
		case part.FunctionResponse != nil:
			if strings.TrimSpace(part.FunctionResponse.ID) == "" {
				continue
			}
			flushMessage()
			raw, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("marshal function response: %w", err)
			}
			items = append(items, map[string]any{
				"type":    "function_call_output",
				"call_id": strings.TrimSpace(part.FunctionResponse.ID),
				"output":  string(raw),
			})
		case part.FunctionCall != nil:
			flushMessage()
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("marshal function call args: %w", err)
			}
			callID := strings.TrimSpace(part.FunctionCall.ID)
			if callID == "" {
				callID = "call_" + uuid.NewString()[:8]
			}
			items = append(items, map[string]any{
				"type":      "function_call",
				"call_id":   callID,
				"name":      part.FunctionCall.Name,
				"arguments": string(argsJSON),
			})
		case part.Text != "":
			contentType := "input_text"
			if role == "assistant" {
				contentType = "output_text"
			}
			messageParts = append(messageParts, map[string]any{
				"type": contentType,
				"text": part.Text,
			})
		case part.InlineData != nil && len(part.InlineData.Data) > 0:
			mimeType := strings.TrimSpace(part.InlineData.MIMEType)
			if strings.HasPrefix(strings.ToLower(mimeType), "image/") && role == "user" {
				messageParts = append(messageParts, map[string]any{
					"type":      "input_image",
					"image_url": "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(part.InlineData.Data),
				})
				continue
			}
			contentType := "input_text"
			if role == "assistant" {
				contentType = "output_text"
			}
			messageParts = append(messageParts, map[string]any{
				"type": contentType,
				"text": string(part.InlineData.Data),
			})
		}
	}

	flushMessage()
	return items, nil
}

func encodeResponsesReasoningSignature(
	provider string,
	modelName string,
	responseModel string,
	item json.RawMessage,
) ([]byte, error) {
	var decoded struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(item, &decoded); err != nil {
		return nil, fmt.Errorf("decode reasoning item: %w", err)
	}
	if decoded.Type != "reasoning" {
		return nil, fmt.Errorf(
			"reasoning item has type %q",
			decoded.Type,
		)
	}
	return json.Marshal(responsesReasoningSignature{
		Version:       responsesReasoningSignatureVersion,
		Provider:      strings.TrimSpace(provider),
		Model:         strings.TrimSpace(modelName),
		ResponseModel: strings.TrimSpace(responseModel),
		Item:          append(json.RawMessage(nil), item...),
	})
}

func decodeResponsesReasoningSignature(
	raw []byte,
	provider string,
	modelName string,
) (json.RawMessage, bool) {
	var signature responsesReasoningSignature
	if err := json.Unmarshal(raw, &signature); err != nil {
		return nil, false
	}
	if signature.Version != responsesReasoningSignatureVersion ||
		!strings.EqualFold(
			strings.TrimSpace(signature.Provider),
			strings.TrimSpace(provider),
		) ||
		strings.TrimSpace(signature.Model) != strings.TrimSpace(modelName) ||
		len(signature.Item) == 0 {
		return nil, false
	}
	if !responsesReplayRouteStable(
		signature.Provider,
		signature.Model,
		signature.ResponseModel,
	) {
		return nil, false
	}
	var item struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(signature.Item, &item); err != nil ||
		item.Type != "reasoning" {
		return nil, false
	}
	return append(json.RawMessage(nil), signature.Item...), true
}

func responsesReplayRouteStable(
	provider string,
	requestModel string,
	responseModel string,
) bool {
	requestModel = strings.TrimSpace(requestModel)
	responseModel = strings.TrimSpace(responseModel)
	if responseModel == "" {
		return false
	}
	if strings.EqualFold(requestModel, responseModel) {
		return true
	}
	provider = strings.ToLower(strings.TrimSpace(provider))
	if provider != "openai" && provider != "codex" {
		return false
	}
	return isDatedModelSnapshot(requestModel, responseModel)
}

func isDatedModelSnapshot(alias string, snapshot string) bool {
	alias = strings.ToLower(strings.TrimSpace(alias))
	snapshot = strings.ToLower(strings.TrimSpace(snapshot))
	prefix := alias + "-"
	if alias == "" || !strings.HasPrefix(snapshot, prefix) {
		return false
	}
	date := strings.TrimPrefix(snapshot, prefix)
	parsed, err := time.Parse("2006-01-02", date)
	return err == nil && parsed.Format("2006-01-02") == date
}

func (m *codexResponsesModel) sendRequest(ctx context.Context, req *codexResponsesRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal codex request: %w", err)
	}

	baseURL := strings.TrimSuffix(m.config.BaseURL, "/")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create codex request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Authorization", "Bearer "+m.config.APIKey)
	httpReq.Header.Set("chatgpt-account-id", m.accountID)
	httpReq.Header.Set("OpenAI-Beta", "responses=experimental")
	httpReq.Header.Set("originator", "ailib")
	if sessionID := clampCodexSessionID(m.config.PromptCacheKey); sessionID != "" {
		httpReq.Header.Set("session-id", sessionID)
		httpReq.Header.Set("x-client-request-id", sessionID)
	}

	httpResp, err := m.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		raw, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, &openAIHTTPStatusError{StatusCode: httpResp.StatusCode, Body: strings.TrimSpace(string(raw))}
	}
	return httpResp, nil
}

func buildCodexFinalResponse(
	acc *responsesAccumulator,
	provider string,
	modelName string,
	usage *genai.GenerateContentResponseUsageMetadata,
	finish genai.FinishReason,
) (*model.LLMResponse, error) {
	parts := make([]*genai.Part, 0, len(acc.outputs)+1)
	emittedCalls := map[*codexResponseCall]struct{}{}
	emittedText := false
	appendCall := func(call *codexResponseCall) error {
		if call == nil || strings.TrimSpace(call.name) == "" {
			return nil
		}
		args := map[string]any{}
		rawArgs := strings.TrimSpace(call.arguments.String())
		if rawArgs != "" {
			if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
				return fmt.Errorf(
					"failed to decode function call arguments for %s: %w",
					call.name,
					err,
				)
			}
		}
		part := genai.NewPartFromFunctionCall(call.name, args)
		part.FunctionCall.ID = firstNonEmptyString(
			call.callID,
			call.itemID,
		)
		parts = append(parts, part)
		emittedCalls[call] = struct{}{}
		return nil
	}
	for _, output := range acc.outputs {
		if output == nil || !output.done {
			continue
		}
		switch output.kind {
		case "reasoning":
			if output.reasoning == nil ||
				len(output.reasoning.rawJSON) == 0 {
				continue
			}
			signature, err := encodeResponsesReasoningSignature(
				provider,
				modelName,
				acc.responseModel,
				output.reasoning.rawJSON,
			)
			if err != nil {
				return nil, err
			}
			parts = append(parts, &genai.Part{
				Text:             output.reasoning.summary,
				Thought:          true,
				ThoughtSignature: signature,
			})
		case "message":
			if output.text.Len() == 0 {
				continue
			}
			parts = append(
				parts,
				genai.NewPartFromText(output.text.String()),
			)
			emittedText = true
		case "image_generation_call":
			if output.image == nil || len(output.image.Data) == 0 {
				continue
			}
			parts = append(parts, &genai.Part{InlineData: output.image})
		case "function_call":
			if err := appendCall(output.call); err != nil {
				return nil, err
			}
		}
	}
	if !emittedText && acc.textBuilder.Len() > 0 {
		parts = append(
			parts,
			genai.NewPartFromText(acc.textBuilder.String()),
		)
	}
	for _, call := range acc.calls {
		if _, ok := emittedCalls[call]; ok {
			continue
		}
		if err := appendCall(call); err != nil {
			return nil, err
		}
	}
	if len(parts) == 0 {
		parts = append(parts, genai.NewPartFromText(" "))
	}
	return &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
		UsageMetadata: usage,
		FinishReason:  finish,
	}, nil
}

func buildCodexUsageMetadata(usage *codexResponsesUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage == nil {
		return nil
	}
	out := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.InputTokens),
		CandidatesTokenCount: int32(usage.OutputTokens),
		TotalTokenCount:      int32(usage.TotalTokens),
	}
	if usage.InputTokensDetails != nil {
		out.CachedContentTokenCount = int32(usage.InputTokensDetails.CachedTokens)
	}
	if out.PromptTokenCount <= 0 && out.CandidatesTokenCount <= 0 && out.TotalTokenCount <= 0 && out.CachedContentTokenCount <= 0 {
		return nil
	}
	return out
}

func (m *codexResponsesModel) newAttempt() providercontract.ModelAttempt {
	provider := "codex"
	if m != nil && m.config != nil && strings.TrimSpace(m.config.Provider) != "" {
		provider = strings.TrimSpace(m.config.Provider)
	}
	baseURL := ""
	if m != nil && m.config != nil {
		baseURL = m.config.BaseURL
	}
	modelName := ""
	if m != nil {
		modelName = m.modelName
	}
	return providercontract.ModelAttempt{
		Provider:     provider,
		Model:        modelName,
		EndpointKind: providercontract.EndpointKindCodexBackendResponses,
		BaseURLClass: providercontract.BaseURLClass(baseURL),
	}
}

func (m *codexResponsesModel) applyErrorAttempt(attempt *providercontract.ModelAttempt, err error) {
	if attempt == nil || err == nil {
		return
	}
	attempt.FailureReason = providercontract.ClassifyError(err)
	attempt.ErrorClass = errorClass(err)
	var statusErr *openAIHTTPStatusError
	if errors.As(err, &statusErr) {
		attempt.StatusCode = statusErr.StatusCode
		attempt.FailureReason = providercontract.ClassifyHTTPStatus(statusErr.StatusCode)
	}
}

func applyCodexSuccessAttempt(
	attempt *providercontract.ModelAttempt,
	usage *genai.GenerateContentResponseUsageMetadata,
	finish genai.FinishReason,
	accountID string,
	responseID string,
) {
	if attempt == nil {
		return
	}
	if attempt.StatusCode == 0 {
		attempt.StatusCode = http.StatusOK
	}
	attempt.FinishReason = string(finish)
	attempt.NativeFinishReason = string(finish)
	attempt.Usage = providercontract.Usage{}
	if usage != nil {
		attempt.Usage.InputTokens = int64(usage.PromptTokenCount)
		attempt.Usage.OutputTokens = int64(usage.CandidatesTokenCount)
		attempt.Usage.TotalTokens = int64(usage.TotalTokenCount)
		if usage.CachedContentTokenCount > 0 {
			attempt.Usage.Cache.ReadTokens = int64(usage.CachedContentTokenCount)
			attempt.Usage.Cache.Hit = true
		}
	}
	attempt.Cache = attempt.Usage.Cache
	attempt.RequestID = strings.TrimSpace(responseID)
	attempt.ProviderRequestID = strings.TrimSpace(responseID)
	attempt.EndpointState.CodexAccountID = strings.TrimSpace(accountID)
}

func (m *codexResponsesModel) observeAttempt(attempt providercontract.ModelAttempt, start time.Time) {
	if m == nil || m.config == nil || m.config.AttemptSink == nil {
		return
	}
	if attempt.LatencyMS == 0 && !start.IsZero() {
		attempt.LatencyMS = time.Since(start).Milliseconds()
	}
	m.config.AttemptSink.ObserveModelAttempt(attempt)
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func defaultCodexInstructions() string {
	return "You are a helpful coding assistant. Follow the user's request and call tools when needed."
}

func buildCodexImageTool(config *genai.GenerateContentConfig) (codexResponsesTool, bool) {
	if !requestsImageResponse(config) {
		return codexResponsesTool{}, false
	}
	tool := codexResponsesTool{
		Type:         "image_generation",
		OutputFormat: "png",
	}
	if config == nil || config.ImageConfig == nil {
		return tool, true
	}
	if size := codexImageSize(config.ImageConfig.AspectRatio); size != "" {
		tool.Size = size
	}
	return tool, true
}

func requestsImageResponse(config *genai.GenerateContentConfig) bool {
	if config == nil {
		return false
	}
	for _, modality := range config.ResponseModalities {
		if strings.EqualFold(strings.TrimSpace(modality), "IMAGE") {
			return true
		}
	}
	return false
}

func hasOnlyCodexImageTool(tools []codexResponsesTool) bool {
	if len(tools) != 1 {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(tools[0].Type), "image_generation")
}

func codexImageSize(aspectRatio string) string {
	switch strings.TrimSpace(aspectRatio) {
	case "1:1":
		return "1024x1024"
	case "3:4", "9:16":
		return "1024x1536"
	case "16:9", "4:3":
		return "1536x1024"
	default:
		return ""
	}
}

func detectCodexImageMimeType(data []byte) string {
	mimeType := strings.TrimSpace(http.DetectContentType(data))
	if strings.HasPrefix(strings.ToLower(mimeType), "image/") {
		return mimeType
	}
	return "image/png"
}

func stripCodexToolMetaSchema(schema map[string]any) map[string]any {
	if len(schema) == 0 {
		return schema
	}
	copied := deepCopySchemaMap(schema)
	props, _ := copied["properties"].(map[string]any)
	if props == nil {
		return copied
	}
	delete(props, "__aos")
	return copied
}

func deepCopySchemaMap(in map[string]any) map[string]any {
	raw, err := json.Marshal(in)
	if err != nil {
		return in
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		return in
	}
	return out
}

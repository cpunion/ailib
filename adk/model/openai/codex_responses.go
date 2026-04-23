package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type codexResponsesModel struct {
	modelName  string
	config     *ClientConfig
	accountID  string
	httpClient *http.Client
}

type codexResponsesRequest struct {
	Model        string                   `json:"model"`
	Store        bool                     `json:"store"`
	Stream       bool                     `json:"stream"`
	Instructions string                   `json:"instructions,omitempty"`
	Input        []any                    `json:"input,omitempty"`
	Tools        []codexResponsesTool     `json:"tools,omitempty"`
	ToolChoice   any                      `json:"tool_choice,omitempty"`
	Reasoning    *codexResponsesReasoning `json:"reasoning,omitempty"`
}

type codexResponsesReasoning struct {
	Effort string `json:"effort"`
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
}

type codexResponsesOutputContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type codexResponsesFinal struct {
	Status            string                 `json:"status"`
	Usage             *codexResponsesUsage   `json:"usage,omitempty"`
	Error             *codexResponsesError   `json:"error,omitempty"`
	IncompleteDetails *codexIncompleteReason `json:"incomplete_details,omitempty"`
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
		httpResp, err := m.sendRequest(ctx, codexReq)
		if err != nil {
			yield(nil, err)
			return
		}
		defer httpResp.Body.Close()

		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4<<20)

		var (
			textBuilder strings.Builder
			calls       []*codexResponseCall
			callsByID   = map[string]*codexResponseCall{}
			images      []*genai.Blob
			seenImages  = map[string]struct{}{}
			usage       *genai.GenerateContentResponseUsageMetadata
			finish      = genai.FinishReasonStop
		)

		ensureCall := func(itemID string) *codexResponseCall {
			if itemID == "" {
				itemID = "call_" + uuid.NewString()[:8]
			}
			if existing := callsByID[itemID]; existing != nil {
				return existing
			}
			call := &codexResponseCall{itemID: itemID}
			callsByID[itemID] = call
			calls = append(calls, call)
			return call
		}

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
				continue
			}

			switch event.Type {
			case "response.output_text.delta":
				if event.Delta == "" {
					continue
				}
				textBuilder.WriteString(event.Delta)
				if stream {
					if !yield(&model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{Text: event.Delta},
							},
						},
						Partial: true,
					}, nil) {
						return
					}
				}
			case "response.function_call_arguments.delta":
				call := ensureCall(event.ItemID)
				call.arguments.WriteString(event.Delta)
			case "response.output_item.added", "response.output_item.done":
				if event.Item == nil {
					continue
				}
				switch event.Item.Type {
				case "function_call":
					call := ensureCall(firstNonEmptyString(event.Item.ID, event.Item.CallID, event.ItemID))
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
					if textBuilder.Len() == 0 {
						for _, part := range event.Item.Content {
							if strings.EqualFold(strings.TrimSpace(part.Type), "output_text") && part.Text != "" {
								textBuilder.WriteString(part.Text)
							}
						}
					}
				case "image_generation_call":
					itemID := firstNonEmptyString(event.Item.ID, event.Item.CallID, event.ItemID)
					if event.Type != "response.output_item.done" || strings.TrimSpace(event.Item.Result) == "" {
						continue
					}
					if _, exists := seenImages[itemID]; exists {
						continue
					}
					imageData, err := base64.StdEncoding.DecodeString(strings.TrimSpace(event.Item.Result))
					if err != nil {
						yield(nil, fmt.Errorf("failed to decode codex image output: %w", err))
						return
					}
					images = append(images, &genai.Blob{
						MIMEType: detectCodexImageMimeType(imageData),
						Data:     imageData,
					})
					seenImages[itemID] = struct{}{}
				}
			case "response.completed":
				if event.Response == nil {
					continue
				}
				usage = buildCodexUsageMetadata(event.Response.Usage)
				switch strings.ToLower(strings.TrimSpace(event.Response.Status)) {
				case "completed", "":
					finish = genai.FinishReasonStop
				case "incomplete":
					if event.Response.IncompleteDetails != nil && strings.EqualFold(strings.TrimSpace(event.Response.IncompleteDetails.Reason), "max_output_tokens") {
						finish = genai.FinishReasonMaxTokens
					} else {
						finish = genai.FinishReasonOther
					}
				default:
					message := "codex response failed"
					if event.Response.Error != nil && strings.TrimSpace(event.Response.Error.Message) != "" {
						message = strings.TrimSpace(event.Response.Error.Message)
					}
					yield(nil, fmt.Errorf("%s", message))
					return
				}
			case "error":
				if event.Response != nil && event.Response.Error != nil && strings.TrimSpace(event.Response.Error.Message) != "" {
					yield(nil, fmt.Errorf("%s", strings.TrimSpace(event.Response.Error.Message)))
					return
				}
				yield(nil, fmt.Errorf("codex responses stream error"))
				return
			}
		}
		if err := scanner.Err(); err != nil {
			yield(nil, fmt.Errorf("codex responses stream error: %w", err))
			return
		}

		finalResp, err := buildCodexFinalResponse(textBuilder.String(), images, calls, usage, finish)
		if err != nil {
			yield(nil, err)
			return
		}
		yield(finalResp, nil)
	}
}

func (m *codexResponsesModel) convertRequest(req *model.LLMRequest) (*codexResponsesRequest, error) {
	out := &codexResponsesRequest{
		Model:  m.modelName,
		Store:  false,
		Stream: true,
	}

	if req.Config != nil && req.Config.SystemInstruction != nil {
		out.Instructions = extractTextFromContent(req.Config.SystemInstruction)
	}
	if strings.TrimSpace(out.Instructions) == "" {
		out.Instructions = defaultCodexInstructions()
	}
	if effort := resolveReasoningEffort("codex", out.Instructions); effort != "" {
		out.Reasoning = &codexResponsesReasoning{Effort: effort}
	}

	for _, content := range req.Contents {
		items, err := convertCodexInputContent(content)
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

func convertCodexInputContent(content *genai.Content) ([]any, error) {
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

	for _, part := range content.Parts {
		switch {
		case part == nil:
			continue
		case part.FunctionResponse != nil:
			if strings.TrimSpace(part.FunctionResponse.ID) == "" {
				continue
			}
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

	if len(messageParts) > 0 {
		items = append([]any{map[string]any{
			"type":    "message",
			"role":    role,
			"content": messageParts,
		}}, items...)
	}
	return items, nil
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

	httpResp, err := m.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		raw, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, fmt.Errorf("API error (status %d): %s", httpResp.StatusCode, strings.TrimSpace(string(raw)))
	}
	return httpResp, nil
}

func buildCodexFinalResponse(text string, images []*genai.Blob, calls []*codexResponseCall, usage *genai.GenerateContentResponseUsageMetadata, finish genai.FinishReason) (*model.LLMResponse, error) {
	parts := make([]*genai.Part, 0, len(images)+len(calls)+1)
	if text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}
	for _, image := range images {
		if image == nil || len(image.Data) == 0 {
			continue
		}
		parts = append(parts, &genai.Part{InlineData: image})
	}
	for _, call := range calls {
		if call == nil || strings.TrimSpace(call.name) == "" {
			continue
		}
		args := map[string]any{}
		rawArgs := strings.TrimSpace(call.arguments.String())
		if rawArgs != "" {
			if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
				return nil, fmt.Errorf("failed to decode function call arguments for %s: %w", call.name, err)
			}
		}
		part := genai.NewPartFromFunctionCall(call.name, args)
		part.FunctionCall.ID = firstNonEmptyString(call.callID, call.itemID)
		parts = append(parts, part)
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

// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package openai implements the [model.LLM] interface for OpenAI-compatible APIs.
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
	"os"
	"strings"
	"time"

	providercontract "github.com/cpunion/ailib/adk/model/provider"
	"github.com/google/uuid"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// ClientConfig holds configuration for the OpenAI client.
type ClientConfig struct {
	// APIKey is the API key for authentication.
	// If empty, will be read from environment variables based on the model name.
	APIKey string
	// BaseURL is the base URL for the API (e.g., "https://api.example.com/v1").
	// If empty, will be inferred from the model name.
	BaseURL string
	// Provider is the logical provider ID (for example openai, openrouter, codex).
	Provider string
	// HTTPClient is the HTTP client to use (optional)
	HTTPClient *http.Client
	// AttemptSink observes normalized provider attempts (optional).
	AttemptSink providercontract.AttemptSink
	// PromptCacheKey is sent as prompt_cache_key on OpenAI-compatible chat
	// completion requests when non-empty.
	PromptCacheKey string
}

// openAIModel implements the model.LLM interface for OpenAI-compatible APIs.
type openAIModel struct {
	modelName  string
	config     *ClientConfig
	httpClient *http.Client
}

// NewModel returns [model.LLM], backed by an OpenAI-compatible API.
//
// It uses the provided context and configuration to initialize the HTTP client.
// The modelName specifies which model to target (e.g., "gpt-4", "gpt-4o-mini").
//
// If config is nil, it will be created with default values.
// If config.APIKey is empty, it will be read from OPENAI_API_KEY environment variable.
// If config.BaseURL is empty, it will be read from OPENAI_BASE_URL environment variable.
//
// An error is returned if no API key or base URL can be found.
func NewModel(ctx context.Context, modelName string, config *ClientConfig) (model.LLM, error) {
	// ctx is reserved for future use (e.g., client initialization with context)
	_ = ctx

	if config == nil {
		config = &ClientConfig{}
	}

	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
		if config.APIKey == "" {
			return nil, fmt.Errorf("openai: API key not found, set OPENAI_API_KEY environment variable or provide config.APIKey")
		}
	}

	if config.BaseURL == "" {
		config.BaseURL = os.Getenv("OPENAI_BASE_URL")
		if config.BaseURL == "" {
			return nil, fmt.Errorf("openai: base URL not found, set OPENAI_BASE_URL environment variable or provide config.BaseURL")
		}
	}

	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	return &openAIModel{
		modelName:  modelName,
		config:     config,
		httpClient: httpClient,
	}, nil
}

// Name returns the model name.
func (m *openAIModel) Name() string {
	return m.modelName
}

// GenerateContent calls the underlying OpenAI-compatible API.
func (m *openAIModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	m.maybeAppendUserContent(req)

	// Convert genai request to OpenAI format
	openaiReq, err := m.convertRequest(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, fmt.Errorf("failed to convert request: %w", err))
		}
	}

	if stream {
		return m.generateStream(ctx, openaiReq)
	}
	return m.generate(ctx, openaiReq)
}

// OpenAI API types
type openAIRequest struct {
	Model           string                `json:"model"`
	Messages        []openAIMessage       `json:"messages"`
	Tools           []openAITool          `json:"tools,omitempty"`
	ReasoningEffort *string               `json:"reasoning_effort,omitempty"`
	Temperature     *float64              `json:"temperature,omitempty"`
	MaxTokens       *int                  `json:"max_tokens,omitempty"`
	TopP            *float64              `json:"top_p,omitempty"`
	Stop            []string              `json:"stop,omitempty"`
	Stream          bool                  `json:"stream,omitempty"`
	ResponseFormat  *openAIResponseFormat `json:"response_format,omitempty"`
	PromptCacheKey  string                `json:"prompt_cache_key,omitempty"`
}

type openAIResponseFormat struct {
	Type string `json:"type"` // "json_object" or "text"
}

type openAIMessage struct {
	Role             string           `json:"role"` // system, user, assistant, tool
	Content          any              `json:"content,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ReasoningContent any              `json:"reasoning_content,omitempty"`
}

type openAIToolCall struct {
	ID       string             `json:"id"`
	Index    *int               `json:"index,omitempty"`
	Type     string             `json:"type"` // "function"
	Function openAIFunctionCall `json:"function"`
}

type openAIFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAITool struct {
	Type     string         `json:"type"` // "function"
	Function openAIFunction `json:"function"`
}

type openAIFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type openAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   *openAIUsage   `json:"usage,omitempty"`
}

type openAIChoice struct {
	Index        int            `json:"index"`
	Message      *openAIMessage `json:"message,omitempty"`
	Delta        *openAIMessage `json:"delta,omitempty"`
	FinishReason string         `json:"finish_reason,omitempty"`
}

type openAIUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *promptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *completionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type completionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// convertRequest converts a model.LLMRequest to OpenAI format
func (m *openAIModel) convertRequest(req *model.LLMRequest) (*openAIRequest, error) {
	openaiReq := &openAIRequest{
		Model:          m.modelName,
		Messages:       make([]openAIMessage, 0),
		PromptCacheKey: strings.TrimSpace(m.config.PromptCacheKey),
	}

	// Add system instruction if present.
	sysContent := ""
	if req.Config != nil && req.Config.SystemInstruction != nil {
		sysContent = extractTextFromContent(req.Config.SystemInstruction)
		if sysContent != "" {
			openaiReq.Messages = append(openaiReq.Messages, openAIMessage{
				Role:    "system",
				Content: sysContent,
			})
		}
	}
	if effort := resolveReasoningEffort(m.config.Provider, sysContent); effort != "" {
		openaiReq.ReasoningEffort = &effort
	}

	// Convert contents to messages
	for _, content := range req.Contents {
		msgs, err := m.convertContent(content)
		if err != nil {
			return nil, fmt.Errorf("failed to convert content: %w", err)
		}
		openaiReq.Messages = append(openaiReq.Messages, msgs...)
	}

	// Convert tools
	if req.Config != nil && len(req.Config.Tools) > 0 {
		for _, tool := range req.Config.Tools {
			if tool.FunctionDeclarations != nil {
				for _, fn := range tool.FunctionDeclarations {
					openaiReq.Tools = append(openaiReq.Tools, convertFunctionDeclaration(fn))
				}
			}
		}
	}

	// Add generation config
	if req.Config != nil {
		if req.Config.Temperature != nil {
			temp := float64(*req.Config.Temperature)
			openaiReq.Temperature = &temp
		}
		if req.Config.MaxOutputTokens > 0 {
			maxTokens := int(req.Config.MaxOutputTokens)
			openaiReq.MaxTokens = &maxTokens
		}
		if req.Config.TopP != nil {
			topP := float64(*req.Config.TopP)
			openaiReq.TopP = &topP
		}
		if len(req.Config.StopSequences) > 0 {
			openaiReq.Stop = req.Config.StopSequences
		}
		if req.Config.ResponseMIMEType == "application/json" {
			openaiReq.ResponseFormat = &openAIResponseFormat{Type: "json_object"}
		}
	}

	return openaiReq, nil
}

func resolveReasoningEffort(provider, systemInstruction string) string {
	if !strings.EqualFold(strings.TrimSpace(provider), "codex") {
		return ""
	}
	level := parseThinkingLevel(systemInstruction)
	switch level {
	case "minimal", "low", "medium", "high", "xhigh":
		return level
	default:
		return ""
	}
}

func parseThinkingLevel(systemInstruction string) string {
	text := strings.TrimSpace(systemInstruction)
	if text == "" {
		return ""
	}
	idx := strings.LastIndex(strings.ToLower(text), "thinking=")
	if idx < 0 {
		return ""
	}
	raw := text[idx+len("thinking="):]
	if cut := strings.IndexAny(raw, " |\n\r\t"); cut >= 0 {
		raw = raw[:cut]
	}
	v := strings.ToLower(strings.TrimSpace(raw))
	switch v {
	case "extra-high", "extra_high", "extrahigh", "x-high", "x_high":
		return "xhigh"
	default:
		return v
	}
}

// convertContent converts genai.Content to OpenAI messages
func (m *openAIModel) convertContent(content *genai.Content) ([]openAIMessage, error) {
	if content == nil || len(content.Parts) == 0 {
		return nil, nil
	}

	role := content.Role
	if role == "model" {
		role = "assistant"
	}

	// Check if this is a tool response
	var toolMessages []openAIMessage
	for _, part := range content.Parts {
		if part.FunctionResponse != nil {
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			toolCallID := part.FunctionResponse.ID
			if toolCallID == "" {
				toolCallID = "call_" + uuid.New().String()[:8]
			}
			toolMessages = append(toolMessages, openAIMessage{
				Role:       "tool",
				Content:    string(responseJSON),
				ToolCallID: toolCallID,
			})
		}
	}
	if len(toolMessages) > 0 {
		return toolMessages, nil
	}

	// Build message content
	var textParts []string
	var contentArray []map[string]any
	var toolCalls []openAIToolCall

	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		} else if part.InlineData != nil && len(part.InlineData.Data) > 0 {
			// Handle inline data (images, video, audio, files, etc.)
			mimeType := part.InlineData.MIMEType
			base64Data := base64.StdEncoding.EncodeToString(part.InlineData.Data)
			dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, base64Data)

			if strings.HasPrefix(mimeType, "image/") {
				contentArray = append(contentArray, map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": dataURI,
					},
				})
			} else if strings.HasPrefix(mimeType, "video/") {
				contentArray = append(contentArray, map[string]any{
					"type": "video_url",
					"video_url": map[string]any{
						"url": dataURI,
					},
				})
			} else if strings.HasPrefix(mimeType, "audio/") {
				contentArray = append(contentArray, map[string]any{
					"type": "audio_url",
					"audio_url": map[string]any{
						"url": dataURI,
					},
				})
			} else if mimeType == "application/pdf" || mimeType == "application/json" {
				contentArray = append(contentArray, map[string]any{
					"type": "file",
					"file": map[string]any{
						"file_data": dataURI,
					},
				})
			} else if strings.HasPrefix(mimeType, "text/") {
				textParts = append(textParts, string(part.InlineData.Data))
			}
		} else if part.FileData != nil && part.FileData.FileURI != "" {
			// Handle file data with URI
			contentArray = append(contentArray, map[string]any{
				"type": "file",
				"file": map[string]any{
					"file_id": part.FileData.FileURI,
				},
			})
		} else if part.FunctionCall != nil {
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function args: %w", err)
			}
			callID := part.FunctionCall.ID
			if callID == "" {
				callID = "call_" + uuid.New().String()[:8]
			}
			toolCalls = append(toolCalls, openAIToolCall{
				ID:   callID,
				Type: "function",
				Function: openAIFunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		}
	}

	msg := openAIMessage{Role: role}

	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		if len(textParts) > 0 {
			msg.Content = strings.Join(textParts, "\n")
		}
	} else if len(contentArray) > 0 {
		// Add text parts to content array
		textMaps := make([]map[string]any, len(textParts))
		for i, text := range textParts {
			textMaps[i] = map[string]any{
				"type": "text",
				"text": text,
			}
		}
		msg.Content = append(textMaps, contentArray...)
	} else if len(textParts) > 0 {
		msg.Content = strings.Join(textParts, "\n")
	}

	return []openAIMessage{msg}, nil
}

// extractTextFromContent extracts and concatenates all text parts from a genai.Content.
func extractTextFromContent(content *genai.Content) string {
	if content == nil {
		return ""
	}
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// convertFunctionDeclaration converts a genai.FunctionDeclaration to OpenAI tool format.
func convertFunctionDeclaration(fn *genai.FunctionDeclaration) openAITool {
	params := convertFunctionParameters(fn)

	return openAITool{
		Type: "function",
		Function: openAIFunction{
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  params,
		},
	}
}

// convertFunctionParameters extracts parameters from a FunctionDeclaration.
// It prefers ParametersJsonSchema (new standard) over Parameters (legacy).
func convertFunctionParameters(fn *genai.FunctionDeclaration) map[string]any {
	// Try ParametersJsonSchema first (new standard used by functiontool)
	if fn.ParametersJsonSchema != nil {
		if params := tryConvertJsonSchema(fn.ParametersJsonSchema); params != nil {
			return params
		}
	}

	// Fallback to Parameters (legacy format used by older code)
	if fn.Parameters != nil {
		return convertLegacyParameters(fn.Parameters)
	}

	return make(map[string]any)
}

// tryConvertJsonSchema attempts to convert ParametersJsonSchema to map[string]any.
// Returns nil if conversion fails.
func tryConvertJsonSchema(schema any) map[string]any {
	// Fast path: already a map
	if params, ok := schema.(map[string]any); ok {
		return params
	}

	// Slow path: convert via JSON marshaling (handles *jsonschema.Schema, etc.)
	jsonBytes, err := json.Marshal(schema)
	if err != nil {
		return nil
	}

	var params map[string]any
	if err := json.Unmarshal(jsonBytes, &params); err != nil {
		return nil
	}

	return params
}

// convertLegacyParameters converts genai.Schema to OpenAI parameters format.
func convertLegacyParameters(schema *genai.Schema) map[string]any {
	params := map[string]any{
		"type": "object",
	}

	if schema.Properties != nil {
		props := make(map[string]any)
		for k, v := range schema.Properties {
			props[k] = schemaToMap(v)
		}
		params["properties"] = props
	}

	if len(schema.Required) > 0 {
		params["required"] = schema.Required
	}

	return params
}

// schemaToMap recursively converts a genai.Schema to a map representation.
func schemaToMap(schema *genai.Schema) map[string]any {
	result := make(map[string]any)
	if schema.Type != genai.TypeUnspecified {
		result["type"] = strings.ToLower(string(schema.Type))
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if schema.Items != nil {
		result["items"] = schemaToMap(schema.Items)
	}
	if schema.Properties != nil {
		props := make(map[string]any)
		for k, v := range schema.Properties {
			props[k] = schemaToMap(v)
		}
		result["properties"] = props
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}
	return result
}

// generate performs a non-streaming API call
func (m *openAIModel) generate(ctx context.Context, openaiReq *openAIRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		start := time.Now()
		attempt := m.newAttempt()
		resp, err := m.doRequest(ctx, openaiReq)
		if err != nil {
			attempt.FailureReason = providercontract.ClassifyError(err)
			attempt.ErrorClass = errorClass(err)
			if statusErr := (*openAIHTTPStatusError)(nil); errors.As(err, &statusErr) {
				attempt.StatusCode = statusErr.StatusCode
				attempt.FailureReason = providercontract.ClassifyHTTPStatus(statusErr.StatusCode)
			}
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		applyOpenAIResponseAttempt(&attempt, resp)

		llmResp, err := m.convertResponse(resp)
		if err != nil {
			attempt.FailureReason = providercontract.ClassifyError(err)
			attempt.ErrorClass = errorClass(err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		if llmResp != nil {
			attempt.FinishReason = string(llmResp.FinishReason)
		}
		m.observeAttempt(attempt, start)
		yield(llmResp, nil)
	}
}

// generateStream performs a streaming API call
func (m *openAIModel) generateStream(ctx context.Context, openaiReq *openAIRequest) iter.Seq2[*model.LLMResponse, error] {
	openaiReq.Stream = true

	return func(yield func(*model.LLMResponse, error) bool) {
		start := time.Now()
		attempt := m.newAttempt()
		httpResp, err := m.sendRequest(ctx, openaiReq)
		if err != nil {
			attempt.FailureReason = providercontract.ClassifyError(err)
			attempt.ErrorClass = errorClass(err)
			if statusErr := (*openAIHTTPStatusError)(nil); errors.As(err, &statusErr) {
				attempt.StatusCode = statusErr.StatusCode
				attempt.FailureReason = providercontract.ClassifyHTTPStatus(statusErr.StatusCode)
			}
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		defer httpResp.Body.Close()
		attempt.StatusCode = httpResp.StatusCode

		scanner := bufio.NewScanner(httpResp.Body)
		var textBuffer strings.Builder
		var toolCalls []openAIToolCall
		var usage *openAIUsage
		var requestID string
		firstTokenAt := time.Time{}
		finishReason := ""

		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}

			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk openAIResponse
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				continue
			}
			attempt.StreamChunkCount++
			if requestID == "" {
				requestID = strings.TrimSpace(chunk.ID)
			}

			// Some OpenAI-compatible providers emit a final usage-only chunk with
			// choices=[]. Record it before inspecting choices.
			if chunk.Usage != nil {
				usage = chunk.Usage
			}

			if len(chunk.Choices) == 0 {
				continue
			}

			choice := chunk.Choices[0]
			delta := choice.Delta
			if delta == nil {
				continue
			}

			// Handle text content
			if delta.Content != nil {
				if text, ok := delta.Content.(string); ok && text != "" {
					if firstTokenAt.IsZero() {
						firstTokenAt = time.Now()
					}
					textBuffer.WriteString(text)
					// Yield partial response
					llmResp := &model.LLMResponse{
						Content: &genai.Content{
							Role: "model",
							Parts: []*genai.Part{
								{Text: text},
							},
						},
						Partial: true,
					}
					if !yield(llmResp, nil) {
						return
					}
				}
			}

			// Handle tool calls
			if len(delta.ToolCalls) > 0 {
				for idx, tc := range delta.ToolCalls {
					targetIdx := idx
					if tc.Index != nil {
						targetIdx = *tc.Index
					}
					// Ensure we have enough space in toolCalls slice
					for len(toolCalls) <= targetIdx {
						toolCalls = append(toolCalls, openAIToolCall{})
					}
					if tc.ID != "" {
						toolCalls[targetIdx].ID = tc.ID
					}
					if tc.Type != "" {
						toolCalls[targetIdx].Type = tc.Type
					}
					if tc.Function.Name != "" {
						toolCalls[targetIdx].Function.Name += tc.Function.Name
					}
					toolCalls[targetIdx].Function.Arguments += tc.Function.Arguments
				}
			}

			// Handle finish
			if choice.FinishReason != "" {
				finishReason = choice.FinishReason
			}
		}

		if err := scanner.Err(); err != nil {
			attempt.FailureReason = providercontract.ClassifyError(err)
			attempt.ErrorClass = errorClass(err)
			attempt.Usage = buildProviderUsage(usage)
			applyRequestCacheKey(&attempt)
			m.observeAttempt(attempt, start)
			yield(nil, fmt.Errorf("stream error: %w", err))
			return
		}

		// Emit final response after [DONE] so usage-only final chunks with
		// choices=[] are not lost.
		if finishReason != "" || textBuffer.Len() > 0 || len(toolCalls) > 0 || usage != nil {
			nativeFinishReason := finishReason
			if nativeFinishReason == "" {
				nativeFinishReason = "stream_eof"
				finishReason = "stop"
			}
			attempt.RequestID = requestID
			attempt.ProviderRequestID = requestID
			attempt.EndpointState.ResponseID = requestID
			attempt.NativeFinishReason = nativeFinishReason
			attempt.FinishReason = string(mapFinishReason(finishReason))
			attempt.Usage = buildProviderUsage(usage)
			applyRequestCacheKey(&attempt)
			if !firstTokenAt.IsZero() {
				attempt.TimeToFirstTokenMS = firstTokenAt.Sub(start).Milliseconds()
			}
			m.observeAttempt(attempt, start)
			finalResp := m.buildFinalResponse(textBuffer.String(), toolCalls, usage, finishReason)
			yield(finalResp, nil)
			return
		}
		attempt.FailureReason = providercontract.FailoverReasonEmpty
		m.observeAttempt(attempt, start)
	}
}

// sendRequest creates and sends an HTTP request to the OpenAI API.
// Caller is responsible for closing the response body.
func (m *openAIModel) sendRequest(ctx context.Context, openaiReq *openAIRequest) (*http.Response, error) {
	reqBody, err := json.Marshal(openaiReq)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	baseURL := strings.TrimSuffix(m.config.BaseURL, "/")
	httpReq, err := http.NewRequestWithContext(ctx, "POST", baseURL+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+m.config.APIKey)

	httpResp, err := m.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(httpResp.Body)
		httpResp.Body.Close()
		return nil, &openAIHTTPStatusError{StatusCode: httpResp.StatusCode, Body: string(body)}
	}

	return httpResp, nil
}

type openAIHTTPStatusError struct {
	StatusCode int
	Body       string
}

func (e *openAIHTTPStatusError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("API error (status %d): %s", e.StatusCode, e.Body)
}

// doRequest performs the HTTP request to the OpenAI API
func (m *openAIModel) doRequest(ctx context.Context, openaiReq *openAIRequest) (*openAIResponse, error) {
	httpResp, err := m.sendRequest(ctx, openaiReq)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()

	var resp openAIResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &resp, nil
}

// convertResponse converts OpenAI response to model.LLMResponse
func (m *openAIModel) convertResponse(resp *openAIResponse) (*model.LLMResponse, error) {
	if len(resp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	choice := resp.Choices[0]
	msg := choice.Message
	if msg == nil {
		return nil, fmt.Errorf("no message in choice")
	}

	var parts []*genai.Part

	// Handle reasoning content (thought process) - prepend before regular content
	if reasoningParts := extractReasoningParts(msg.ReasoningContent); len(reasoningParts) > 0 {
		parts = append(parts, reasoningParts...)
	}

	// Get tool calls - either from structured response or parsed from text
	toolCalls := msg.ToolCalls
	textContent := ""
	if msg.Content != nil {
		if text, ok := msg.Content.(string); ok {
			textContent = text
		}
	}

	// If no structured tool calls, try parsing from text content
	if len(toolCalls) == 0 && textContent != "" {
		parsedCalls, remainder := parseToolCallsFromText(textContent)
		if len(parsedCalls) > 0 {
			toolCalls = parsedCalls
			textContent = remainder
		}
	}

	// Handle text content
	if textContent != "" {
		parts = append(parts, genai.NewPartFromText(textContent))
	}

	// Handle tool calls
	for _, tc := range toolCalls {
		if tc.ID == "" && tc.Function.Name == "" && tc.Function.Arguments == "" {
			continue
		}
		var args map[string]any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			return nil, fmt.Errorf("failed to unmarshal tool arguments: %w", err)
		}
		part := genai.NewPartFromFunctionCall(tc.Function.Name, args)
		part.FunctionCall.ID = tc.ID
		parts = append(parts, part)
	}

	// Ensure at least one part exists (ADK requires non-empty parts)
	if len(parts) == 0 {
		parts = append(parts, genai.NewPartFromText(" "))
	}

	llmResp := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
	}

	// Add usage metadata
	llmResp.UsageMetadata = buildUsageMetadata(resp.Usage)

	// Map finish reason
	llmResp.FinishReason = mapFinishReason(choice.FinishReason)

	return llmResp, nil
}

func (m *openAIModel) buildFinalResponse(text string, toolCalls []openAIToolCall, usage *openAIUsage, finishReason string) *model.LLMResponse {
	// Ensure at least one part exists (ADK requires non-empty parts)
	if text == "" && len(toolCalls) == 0 {
		text = " "
	}
	var parts []*genai.Part

	if text != "" {
		parts = append(parts, genai.NewPartFromText(text))
	}

	for _, tc := range toolCalls {
		if tc.ID == "" && tc.Function.Name == "" && tc.Function.Arguments == "" {
			continue
		}
		var args map[string]any
		if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
			continue
		}
		part := genai.NewPartFromFunctionCall(tc.Function.Name, args)
		part.FunctionCall.ID = tc.ID
		parts = append(parts, part)
	}

	// Ensure at least one part exists (ADK requires non-empty parts)
	if len(parts) == 0 {
		parts = append(parts, genai.NewPartFromText(" "))
	}

	llmResp := &model.LLMResponse{
		Content: &genai.Content{
			Role:  "model",
			Parts: parts,
		},
		FinishReason:  mapFinishReason(finishReason),
		UsageMetadata: buildUsageMetadata(usage),
	}

	return llmResp
}

// buildUsageMetadata converts OpenAI usage data to genai usage metadata.
func buildUsageMetadata(usage *openAIUsage) *genai.GenerateContentResponseUsageMetadata {
	if usage == nil {
		return nil
	}
	metadata := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(usage.PromptTokens),
		CandidatesTokenCount: int32(usage.CompletionTokens),
		TotalTokenCount:      int32(usage.TotalTokens),
	}
	// Add cached token count if available
	if usage.PromptTokensDetails != nil {
		metadata.CachedContentTokenCount = int32(usage.PromptTokensDetails.CachedTokens)
	}
	return metadata
}

func buildProviderUsage(usage *openAIUsage) providercontract.Usage {
	if usage == nil {
		return providercontract.Usage{}
	}
	out := providercontract.Usage{
		InputTokens:  int64(usage.PromptTokens),
		OutputTokens: int64(usage.CompletionTokens),
		TotalTokens:  int64(usage.TotalTokens),
	}
	if usage.PromptTokensDetails != nil {
		out.Cache.ReadTokens = int64(usage.PromptTokensDetails.CachedTokens)
		out.Cache.Hit = usage.PromptTokensDetails.CachedTokens > 0
	}
	if usage.CompletionTokensDetails != nil {
		out.ReasoningTokens = int64(usage.CompletionTokensDetails.ReasoningTokens)
	}
	return out
}

func (m *openAIModel) newAttempt() providercontract.ModelAttempt {
	provider := strings.TrimSpace(m.config.Provider)
	if provider == "" {
		provider = "openai"
	}
	return providercontract.ModelAttempt{
		Provider:     provider,
		Model:        m.modelName,
		EndpointKind: providercontract.EndpointKindChatCompletions,
		BaseURLClass: providercontract.BaseURLClass(m.config.BaseURL),
		Cache: providercontract.CacheUsage{
			CacheKey: strings.TrimSpace(m.config.PromptCacheKey),
		},
	}
}

func (m *openAIModel) observeAttempt(attempt providercontract.ModelAttempt, start time.Time) {
	if m == nil || m.config == nil || m.config.AttemptSink == nil {
		return
	}
	if attempt.LatencyMS == 0 && !start.IsZero() {
		attempt.LatencyMS = time.Since(start).Milliseconds()
	}
	m.config.AttemptSink.ObserveModelAttempt(attempt)
}

func applyOpenAIResponseAttempt(attempt *providercontract.ModelAttempt, resp *openAIResponse) {
	if attempt == nil || resp == nil {
		return
	}
	attempt.StatusCode = http.StatusOK
	attempt.RequestID = strings.TrimSpace(resp.ID)
	attempt.ProviderRequestID = strings.TrimSpace(resp.ID)
	attempt.EndpointState.ResponseID = strings.TrimSpace(resp.ID)
	attempt.Usage = buildProviderUsage(resp.Usage)
	applyRequestCacheKey(attempt)
	if len(resp.Choices) > 0 {
		attempt.NativeFinishReason = resp.Choices[0].FinishReason
		attempt.FinishReason = string(mapFinishReason(resp.Choices[0].FinishReason))
	}
}

func applyRequestCacheKey(attempt *providercontract.ModelAttempt) {
	if attempt == nil {
		return
	}
	if key := strings.TrimSpace(attempt.Cache.CacheKey); key != "" {
		attempt.Usage.Cache.CacheKey = key
	}
	attempt.Cache = attempt.Usage.Cache
}

func errorClass(err error) string {
	if err == nil {
		return ""
	}
	if statusErr := (*openAIHTTPStatusError)(nil); errors.As(err, &statusErr) {
		return "http_status"
	}
	if errors.Is(err, context.Canceled) {
		return "context_canceled"
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return "deadline_exceeded"
	}
	name := fmt.Sprintf("%T", err)
	if strings.HasPrefix(name, "*") {
		name = strings.TrimPrefix(name, "*")
	}
	return name
}

// extractReasoningParts extracts reasoning/thought content from provider-specific payloads.
// It converts various reasoning formats (string, list, map) into genai.Part with Thought=true.
func extractReasoningParts(reasoningContent any) []*genai.Part {
	if reasoningContent == nil {
		return nil
	}

	var parts []*genai.Part
	extractTexts(reasoningContent, &parts)
	return parts
}

// extractTexts recursively extracts text from reasoning content and creates thought parts.
func extractTexts(value any, parts *[]*genai.Part) {
	if value == nil {
		return
	}

	switch v := value.(type) {
	case string:
		if v != "" {
			*parts = append(*parts, &genai.Part{Text: v, Thought: true})
		}
	case []any:
		for _, item := range v {
			extractTexts(item, parts)
		}
	case map[string]any:
		// LiteLLM/OpenAI nests reasoning text under known keys
		for _, key := range []string{"text", "content", "reasoning", "reasoning_content"} {
			if text, ok := v[key].(string); ok && text != "" {
				*parts = append(*parts, &genai.Part{Text: text, Thought: true})
			}
		}
	}
}

// parseToolCallsFromText extracts inline JSON tool calls from text responses.
// Some models embed tool calls as JSON objects in their text output.
// Returns the extracted tool calls and any remaining text.
func parseToolCallsFromText(text string) ([]openAIToolCall, string) {
	if text == "" {
		return nil, ""
	}

	var toolCalls []openAIToolCall
	var remainder strings.Builder
	cursor := 0

	for cursor < len(text) {
		braceIndex := strings.Index(text[cursor:], "{")
		if braceIndex == -1 {
			remainder.WriteString(text[cursor:])
			break
		}
		braceIndex += cursor

		remainder.WriteString(text[cursor:braceIndex])

		// Try to parse JSON starting at brace
		var candidate map[string]any
		decoder := json.NewDecoder(strings.NewReader(text[braceIndex:]))
		if err := decoder.Decode(&candidate); err != nil {
			remainder.WriteString(text[braceIndex : braceIndex+1])
			cursor = braceIndex + 1
			continue
		}

		// Calculate end position
		endPos := braceIndex + int(decoder.InputOffset())

		// Check if this looks like a tool call
		name, hasName := candidate["name"].(string)
		args, hasArgs := candidate["arguments"]
		if hasName && hasArgs {
			argsStr := ""
			switch a := args.(type) {
			case string:
				argsStr = a
			default:
				if jsonBytes, err := json.Marshal(args); err == nil {
					argsStr = string(jsonBytes)
				}
			}

			callID := "call_" + uuid.New().String()[:8]
			if id, ok := candidate["id"].(string); ok && id != "" {
				callID = id
			}

			toolCalls = append(toolCalls, openAIToolCall{
				ID:   callID,
				Type: "function",
				Function: openAIFunctionCall{
					Name:      name,
					Arguments: argsStr,
				},
			})
		} else {
			remainder.WriteString(text[braceIndex:endPos])
		}
		cursor = endPos
	}

	return toolCalls, strings.TrimSpace(remainder.String())
}

// mapFinishReason maps OpenAI finish_reason strings to genai.FinishReason values.
// Note: tool_calls and function_call map to STOP because tool calls represent
// normal completion where the model stopped to invoke tools.
func mapFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "stop":
		return genai.FinishReasonStop
	case "length":
		return genai.FinishReasonMaxTokens
	case "tool_calls", "function_call":
		return genai.FinishReasonStop
	case "content_filter":
		return genai.FinishReasonSafety
	default:
		return genai.FinishReasonOther
	}
}

// maybeAppendUserContent appends a user content, so that model can continue to output.
func (m *openAIModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
		return
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

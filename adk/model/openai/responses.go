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

package openai

import (
	"bufio"
	"bytes"
	"context"
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
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// responsesModel implements model.LLM against the generic OpenAI Responses API
// (POST {baseURL}/responses). It shares the SSE event accumulator and the
// Responses input/output conversion with the Codex backend variant; the
// difference is transport (no chatgpt-account-id header), the endpoint-kind
// attribution, and that it carries prompt_cache_key / sampling options through.
type responsesModel struct {
	modelName  string
	config     *ClientConfig
	httpClient *http.Client
}

// responsesRequest is the generic /v1/responses request body. Codex uses its
// own struct (store/account semantics differ); the field shapes that overlap
// reuse the same JSON contract.
type responsesRequest struct {
	Model           string                   `json:"model"`
	Stream          bool                     `json:"stream"`
	Instructions    string                   `json:"instructions,omitempty"`
	Input           []any                    `json:"input,omitempty"`
	Tools           []codexResponsesTool     `json:"tools,omitempty"`
	ToolChoice      any                      `json:"tool_choice,omitempty"`
	Reasoning       *codexResponsesReasoning `json:"reasoning,omitempty"`
	PromptCacheKey  string                   `json:"prompt_cache_key,omitempty"`
	MaxOutputTokens *int                     `json:"max_output_tokens,omitempty"`
	Temperature     *float64                 `json:"temperature,omitempty"`
	TopP            *float64                 `json:"top_p,omitempty"`
}

// NewResponsesModel returns a model.LLM backed by the OpenAI Responses API.
//
// Like NewModel, a nil config is filled from OPENAI_API_KEY / OPENAI_BASE_URL,
// and an error is returned when neither config nor environment supplies them.
func NewResponsesModel(ctx context.Context, modelName string, config *ClientConfig) (model.LLM, error) {
	_ = ctx
	if config == nil {
		config = &ClientConfig{}
	}
	if config.APIKey == "" {
		config.APIKey = os.Getenv("OPENAI_API_KEY")
		if config.APIKey == "" {
			return nil, fmt.Errorf("openai responses: API key not found, set OPENAI_API_KEY or provide config.APIKey")
		}
	}
	if config.BaseURL == "" {
		config.BaseURL = os.Getenv("OPENAI_BASE_URL")
		if config.BaseURL == "" {
			return nil, fmt.Errorf("openai responses: base URL not found, set OPENAI_BASE_URL or provide config.BaseURL")
		}
	}
	httpClient := config.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	return &responsesModel{modelName: modelName, config: config, httpClient: httpClient}, nil
}

func (m *responsesModel) Name() string { return m.modelName }

func (m *responsesModel) provider() string {
	if m != nil && m.config != nil && strings.TrimSpace(m.config.Provider) != "" {
		return strings.TrimSpace(m.config.Provider)
	}
	return "openai"
}

func (m *responsesModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	(&openAIModel{}).maybeAppendUserContent(req)
	r, err := m.convertRequest(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, fmt.Errorf("failed to convert responses request: %w", err))
		}
	}
	// Request-scoped prompt cache key wins over the client-level default:
	// clients are shared across sessions, cache bucketing is per session.
	if key := providercontract.PromptCacheKeyFromContext(ctx); key != "" {
		r.PromptCacheKey = key
	} else if key := strings.TrimSpace(m.config.PromptCacheKey); key != "" {
		r.PromptCacheKey = key
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		start := time.Now()
		attempt := m.newAttempt()
		httpResp, err := m.sendRequest(ctx, r)
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
				continue
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
			err = fmt.Errorf("openai responses stream error: %w", err)
			m.applyErrorAttempt(&attempt, err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}

		finalResp, err := buildCodexFinalResponse(acc.textBuilder.String(), acc.images, acc.calls, acc.usage, acc.finish)
		if err != nil {
			m.applyErrorAttempt(&attempt, err)
			m.observeAttempt(attempt, start)
			yield(nil, err)
			return
		}
		m.applySuccessAttempt(&attempt, acc.usage, acc.finish)
		m.observeAttempt(attempt, start)
		yield(finalResp, nil)
	}
}

func (m *responsesModel) convertRequest(req *model.LLMRequest) (*responsesRequest, error) {
	out := &responsesRequest{Model: m.modelName, Stream: true}

	if req.Config != nil && req.Config.SystemInstruction != nil {
		out.Instructions = extractTextFromContent(req.Config.SystemInstruction)
	}
	if effort := parseThinkingLevel(out.Instructions); effort != "" {
		out.Reasoning = &codexResponsesReasoning{Effort: effort}
	}

	for _, content := range req.Contents {
		items, err := convertResponsesInputContent(content)
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
		if len(out.Tools) > 0 {
			out.ToolChoice = "auto"
		}
		if req.Config.Temperature != nil {
			temp := float64(*req.Config.Temperature)
			out.Temperature = &temp
		}
		if req.Config.TopP != nil {
			topP := float64(*req.Config.TopP)
			out.TopP = &topP
		}
		if req.Config.MaxOutputTokens > 0 {
			maxTokens := int(req.Config.MaxOutputTokens)
			out.MaxOutputTokens = &maxTokens
		}
	}
	return out, nil
}

func (m *responsesModel) sendRequest(ctx context.Context, req *responsesRequest) (*http.Response, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal responses request: %w", err)
	}
	baseURL := strings.TrimSuffix(m.config.BaseURL, "/")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/responses", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create responses request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	httpReq.Header.Set("Authorization", "Bearer "+m.config.APIKey)

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

func (m *responsesModel) newAttempt() providercontract.ModelAttempt {
	baseURL := ""
	if m != nil && m.config != nil {
		baseURL = m.config.BaseURL
	}
	modelName := ""
	if m != nil {
		modelName = m.modelName
	}
	return providercontract.ModelAttempt{
		Provider:     m.provider(),
		Model:        modelName,
		EndpointKind: providercontract.EndpointKindResponses,
		BaseURLClass: providercontract.BaseURLClass(baseURL),
	}
}

func (m *responsesModel) applyErrorAttempt(attempt *providercontract.ModelAttempt, err error) {
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

func (m *responsesModel) applySuccessAttempt(attempt *providercontract.ModelAttempt, usage *genai.GenerateContentResponseUsageMetadata, finish genai.FinishReason) {
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
}

func (m *responsesModel) observeAttempt(attempt providercontract.ModelAttempt, start time.Time) {
	if m == nil || m.config == nil || m.config.AttemptSink == nil {
		return
	}
	if attempt.LatencyMS == 0 && !start.IsZero() {
		attempt.LatencyMS = time.Since(start).Milliseconds()
	}
	m.config.AttemptSink.ObserveModelAttempt(attempt)
}

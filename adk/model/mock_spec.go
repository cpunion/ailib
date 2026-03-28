package model

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"os"
	"strings"
	"sync"
	"time"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

type MockUsage struct {
	PromptTokens     int64 `json:"promptTokens,omitempty"`
	CompletionTokens int64 `json:"completionTokens,omitempty"`
	TotalTokens      int64 `json:"totalTokens,omitempty"`
}

type MockFunctionCallSpec struct {
	ID   string         `json:"id,omitempty"`
	Name string         `json:"name,omitempty"`
	Args map[string]any `json:"args,omitempty"`
}

type MockFunctionResponseSpec struct {
	ID       string         `json:"id,omitempty"`
	Name     string         `json:"name,omitempty"`
	Response map[string]any `json:"response,omitempty"`
}

type MockPartSpec struct {
	Text             string                    `json:"text,omitempty"`
	Thought          bool                      `json:"thought,omitempty"`
	FunctionCall     *MockFunctionCallSpec     `json:"functionCall,omitempty"`
	FunctionResponse *MockFunctionResponseSpec `json:"functionResponse,omitempty"`
}

type MockResponseSpec struct {
	Text         string         `json:"text,omitempty"`
	Parts        []MockPartSpec `json:"parts,omitempty"`
	Partial      bool           `json:"partial,omitempty"`
	TurnComplete bool           `json:"turnComplete,omitempty"`
	Interrupted  bool           `json:"interrupted,omitempty"`
	ErrorMessage string         `json:"errorMessage,omitempty"`
	ErrorCode    string         `json:"errorCode,omitempty"`
	Usage        *MockUsage     `json:"usage,omitempty"`
}

type MockTurnSpec struct {
	Text                         string             `json:"text,omitempty"`
	Stream                       bool               `json:"stream,omitempty"`
	ChunkDelayMs                 int                `json:"chunkDelayMs,omitempty"`
	Chunks                       []string           `json:"chunks,omitempty"`
	Final                        string             `json:"final,omitempty"`
	Usage                        *MockUsage         `json:"usage,omitempty"`
	Responses                    []MockResponseSpec `json:"responses,omitempty"`
	ExpectLatestUserText         string             `json:"expectLatestUserText,omitempty"`
	ExpectFunctionResponseName   string             `json:"expectFunctionResponseName,omitempty"`
	ExpectFunctionResponseID     string             `json:"expectFunctionResponseId,omitempty"`
	ExpectFunctionResponseStatus string             `json:"expectFunctionResponseStatus,omitempty"`
}

// MockSpec configures deterministic mock-provider behavior for tests and local development.
type MockSpec struct {
	Text         string             `json:"text,omitempty"`
	Stream       bool               `json:"stream,omitempty"`
	ChunkDelayMs int                `json:"chunkDelayMs,omitempty"`
	Chunks       []string           `json:"chunks,omitempty"`
	Final        string             `json:"final,omitempty"`
	Usage        *MockUsage         `json:"usage,omitempty"`
	Responses    []MockResponseSpec `json:"responses,omitempty"`
	Cases        []MockTurnSpec     `json:"cases,omitempty"`
	Turns        []MockTurnSpec     `json:"turns,omitempty"`
}

// ResolveMockSpec resolves inline JSON or @file references used by the mock provider.
func ResolveMockSpec(modelName string) (string, error) {
	v := strings.TrimSpace(modelName)
	if strings.HasPrefix(v, "@") {
		path := strings.TrimSpace(strings.TrimPrefix(v, "@"))
		if path == "" {
			return "", errors.New("mock spec file path required after '@'")
		}
		raw, err := os.ReadFile(path)
		if err != nil {
			return "", err
		}
		return string(raw), nil
	}
	return v, nil
}

type MockSpecLLM struct {
	Spec MockSpec

	mu      sync.Mutex
	current int
}

func NewMockSpecLLM(spec MockSpec) *MockSpecLLM {
	return &MockSpecLLM{Spec: spec}
}

func (m *MockSpecLLM) Name() string {
	return "mock-spec"
}

func (m *MockSpecLLM) GenerateContent(ctx context.Context, req *adkmodel.LLMRequest, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	spec := MockSpec{}
	if m != nil {
		spec = m.Spec
	}
	if turnSpec, ok, err := selectMockCase(spec, req); ok || err != nil {
		if err != nil {
			return func(yield func(*adkmodel.LLMResponse, error) bool) {
				yield(nil, err)
			}
		}
		return emitMockTurn(ctx, turnSpec, stream)
	}
	turnSpec, ok, err := m.nextTurn(spec, req)
	if err != nil {
		return func(yield func(*adkmodel.LLMResponse, error) bool) {
			yield(nil, err)
		}
	}
	if !ok {
		return func(yield func(*adkmodel.LLMResponse, error) bool) {
			yield(nil, errors.New("no scripted mock turn available"))
		}
	}
	return emitMockTurn(ctx, turnSpec, stream)
}

func emitMockTurn(ctx context.Context, turnSpec MockTurnSpec, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	if len(turnSpec.Responses) > 0 {
		return emitScriptedResponses(turnSpec, stream)
	}
	delay := time.Duration(turnSpec.ChunkDelayMs) * time.Millisecond
	if delay < 0 {
		delay = 0
	}
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		if stream && turnSpec.Stream && len(turnSpec.Chunks) > 0 {
			acc := strings.Builder{}
			for _, chunk := range turnSpec.Chunks {
				if ctx.Err() != nil {
					finalText := acc.String()
					if finalText == "" {
						finalText = "INTERRUPTED"
					}
					yield(&adkmodel.LLMResponse{
						Content:        genai.NewContentFromText(finalText, genai.RoleModel),
						Partial:        false,
						TurnComplete:   true,
						Interrupted:    true,
						ErrorMessage:   ctx.Err().Error(),
						ErrorCode:      "context_canceled",
						FinishReason:   genai.FinishReasonStop,
						AvgLogprobs:    0,
						LogprobsResult: nil,
					}, nil)
					return
				}
				if delay > 0 {
					t := time.NewTimer(delay)
					select {
					case <-t.C:
					case <-ctx.Done():
						if !t.Stop() {
							<-t.C
						}
						continue
					}
				}
				acc.WriteString(chunk)
				if !yield(&adkmodel.LLMResponse{
					Content: genai.NewContentFromText(acc.String(), genai.RoleModel),
					Partial: true,
				}, nil) {
					return
				}
			}
			finalText := strings.TrimSpace(turnSpec.Final)
			if finalText == "" {
				finalText = acc.String()
			}
			resp := &adkmodel.LLMResponse{
				Content:      genai.NewContentFromText(finalText, genai.RoleModel),
				Partial:      false,
				TurnComplete: true,
			}
			if turnSpec.Usage != nil {
				resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     int32(turnSpec.Usage.PromptTokens),
					CandidatesTokenCount: int32(turnSpec.Usage.CompletionTokens),
					TotalTokenCount:      int32(turnSpec.Usage.TotalTokens),
				}
			}
			yield(resp, nil)
			return
		}

		text := strings.TrimSpace(turnSpec.Text)
		if text == "" {
			text = strings.TrimSpace(turnSpec.Final)
		}
		if text == "" && len(turnSpec.Chunks) > 0 {
			text = strings.Join(turnSpec.Chunks, "")
		}
		if text == "" {
			text = "MOCK_OK"
		}
		resp := &adkmodel.LLMResponse{
			Content:      genai.NewContentFromText(text, genai.RoleModel),
			Partial:      false,
			TurnComplete: true,
		}
		if turnSpec.Usage != nil {
			resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     int32(turnSpec.Usage.PromptTokens),
				CandidatesTokenCount: int32(turnSpec.Usage.CompletionTokens),
				TotalTokenCount:      int32(turnSpec.Usage.TotalTokens),
			}
		}
		yield(resp, nil)
	}
}

func (m *MockSpecLLM) nextTurn(spec MockSpec, req *adkmodel.LLMRequest) (MockTurnSpec, bool, error) {
	if len(spec.Turns) == 0 {
		return MockTurnSpec{
			Text:         spec.Text,
			Stream:       spec.Stream,
			ChunkDelayMs: spec.ChunkDelayMs,
			Chunks:       append([]string(nil), spec.Chunks...),
			Final:        spec.Final,
			Usage:        spec.Usage,
			Responses:    append([]MockResponseSpec(nil), spec.Responses...),
		}, true, nil
	}
	if m == nil {
		return MockTurnSpec{}, false, nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.current >= len(spec.Turns) {
		return MockTurnSpec{}, false, nil
	}
	turn := spec.Turns[m.current]
	if err := validateMockTurnRequest(turn, req); err != nil {
		return MockTurnSpec{}, false, fmt.Errorf("mock turn %d request mismatch: %w", m.current, err)
	}
	m.current++
	return turn, true, nil
}

func selectMockCase(spec MockSpec, req *adkmodel.LLMRequest) (MockTurnSpec, bool, error) {
	if len(spec.Cases) == 0 {
		return MockTurnSpec{}, false, nil
	}
	for _, item := range spec.Cases {
		if mockTurnMatches(item, req) {
			return item, true, nil
		}
	}
	if hasDefaultMockFallback(spec) {
		return MockTurnSpec{}, false, nil
	}
	return MockTurnSpec{}, false, fmt.Errorf("no matching mock case for latest user text %q", latestUserText(req))
}

func mockTurnMatches(turn MockTurnSpec, req *adkmodel.LLMRequest) bool {
	return validateMockTurnRequest(turn, req) == nil
}

func hasDefaultMockFallback(spec MockSpec) bool {
	return len(spec.Turns) > 0 ||
		len(spec.Responses) > 0 ||
		strings.TrimSpace(spec.Text) != "" ||
		strings.TrimSpace(spec.Final) != "" ||
		len(spec.Chunks) > 0
}

type mockObservedFunctionResponse struct {
	ID     string
	Name   string
	Status string
}

func validateMockTurnRequest(turn MockTurnSpec, req *adkmodel.LLMRequest) error {
	if want := strings.TrimSpace(turn.ExpectLatestUserText); want != "" {
		got := strings.TrimSpace(latestUserText(req))
		if !strings.Contains(strings.ToLower(got), strings.ToLower(want)) {
			return fmt.Errorf("expected latest user text containing %q, got %q", want, got)
		}
	}
	if strings.TrimSpace(turn.ExpectFunctionResponseName) == "" &&
		strings.TrimSpace(turn.ExpectFunctionResponseID) == "" &&
		strings.TrimSpace(turn.ExpectFunctionResponseStatus) == "" {
		return nil
	}
	got, ok := latestFunctionResponse(req)
	if !ok {
		return errors.New("expected latest functionResponse in request, got none")
	}
	if want := strings.TrimSpace(turn.ExpectFunctionResponseName); want != "" && !strings.EqualFold(strings.TrimSpace(got.Name), want) {
		return fmt.Errorf("expected functionResponse name %q, got %q", want, got.Name)
	}
	if want := strings.TrimSpace(turn.ExpectFunctionResponseID); want != "" && strings.TrimSpace(got.ID) != want {
		return fmt.Errorf("expected functionResponse id %q, got %q", want, got.ID)
	}
	if want := strings.TrimSpace(turn.ExpectFunctionResponseStatus); want != "" && !strings.EqualFold(strings.TrimSpace(got.Status), want) {
		return fmt.Errorf("expected functionResponse status %q, got %q", want, got.Status)
	}
	return nil
}

func latestFunctionResponse(req *adkmodel.LLMRequest) (mockObservedFunctionResponse, bool) {
	if req == nil || len(req.Contents) == 0 {
		return mockObservedFunctionResponse{}, false
	}
	for i := len(req.Contents) - 1; i >= 0; i-- {
		c := req.Contents[i]
		if c == nil || len(c.Parts) == 0 {
			continue
		}
		for j := len(c.Parts) - 1; j >= 0; j-- {
			p := c.Parts[j]
			if p == nil || p.FunctionResponse == nil {
				continue
			}
			resp := mockObservedFunctionResponse{
				ID:   strings.TrimSpace(p.FunctionResponse.ID),
				Name: strings.TrimSpace(p.FunctionResponse.Name),
			}
			if status, _ := p.FunctionResponse.Response["status"].(string); strings.TrimSpace(status) != "" {
				resp.Status = strings.TrimSpace(status)
			}
			return resp, true
		}
	}
	return mockObservedFunctionResponse{}, false
}

func emitScriptedResponses(turn MockTurnSpec, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		if len(turn.Responses) == 0 {
			return
		}
		if stream {
			for _, spec := range turn.Responses {
				if !yield(mockResponseFromSpec(spec), nil) {
					return
				}
			}
			return
		}
		yield(mockResponseFromSpec(turn.Responses[len(turn.Responses)-1]), nil)
	}
}

func mockResponseFromSpec(spec MockResponseSpec) *adkmodel.LLMResponse {
	parts := make([]*genai.Part, 0, max(1, len(spec.Parts)))
	if len(spec.Parts) > 0 {
		for _, part := range spec.Parts {
			if p := mockPartFromSpec(part); p != nil {
				parts = append(parts, p)
			}
		}
	} else if strings.TrimSpace(spec.Text) != "" {
		parts = append(parts, genai.NewPartFromText(spec.Text))
	}
	if len(parts) == 0 {
		parts = append(parts, genai.NewPartFromText("MOCK_OK"))
	}
	resp := &adkmodel.LLMResponse{
		Content:      genai.NewContentFromParts(parts, genai.RoleModel),
		Partial:      spec.Partial,
		TurnComplete: spec.TurnComplete,
		Interrupted:  spec.Interrupted,
		ErrorMessage: spec.ErrorMessage,
		ErrorCode:    spec.ErrorCode,
	}
	if spec.Usage != nil {
		resp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(spec.Usage.PromptTokens),
			CandidatesTokenCount: int32(spec.Usage.CompletionTokens),
			TotalTokenCount:      int32(spec.Usage.TotalTokens),
		}
	}
	return resp
}

func mockPartFromSpec(spec MockPartSpec) *genai.Part {
	switch {
	case spec.FunctionCall != nil:
		part := genai.NewPartFromFunctionCall(spec.FunctionCall.Name, spec.FunctionCall.Args)
		if part.FunctionCall != nil {
			part.FunctionCall.ID = spec.FunctionCall.ID
		}
		return part
	case spec.FunctionResponse != nil:
		part := genai.NewPartFromFunctionResponse(spec.FunctionResponse.Name, spec.FunctionResponse.Response)
		if part.FunctionResponse != nil {
			part.FunctionResponse.ID = spec.FunctionResponse.ID
		}
		return part
	case spec.Text != "" || spec.Thought:
		part := genai.NewPartFromText(spec.Text)
		part.Thought = spec.Thought
		return part
	default:
		return nil
	}
}

func (m *MockSpecLLM) Reset() {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.current = 0
}

var _ adkmodel.LLM = (*MockSpecLLM)(nil)

package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	providercontract "github.com/cpunion/ailib/adk/model/provider"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestResponsesModelGenerateText(t *testing.T) {
	var (
		seenPath    string
		seenAuth    string
		seenAccount string
		seenModel   string
		seenStream  bool
		seenPrompt  string
		seenCache   string
		seenInput   []any
		attempts    []providercontract.ModelAttempt
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Model          string `json:"model"`
			Stream         bool   `json:"stream"`
			Instructions   string `json:"instructions"`
			PromptCacheKey string `json:"prompt_cache_key"`
			Input          []any  `json:"input"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		seenPath = r.URL.Path
		seenAuth = r.Header.Get("Authorization")
		seenAccount = r.Header.Get("chatgpt-account-id")
		seenModel = req.Model
		seenStream = req.Stream
		seenPrompt = req.Instructions
		seenCache = req.PromptCacheKey
		seenInput = req.Input

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\" world\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":12,\"output_tokens\":5,\"total_tokens\":17,\"input_tokens_details\":{\"cached_tokens\":4}}}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey:         "sk-test",
		BaseURL:        ts.URL,
		HTTPClient:     ts.Client(),
		Provider:       "openai",
		PromptCacheKey: "sess-cache-key",
		AttemptSink: providercontract.AttemptSinkFunc(func(a providercontract.ModelAttempt) {
			attempts = append(attempts, a)
		}),
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Say hello.", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are helpful.", "system"),
		},
	}
	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	if got := extractPartsText(responses[0].Content); got != "Hello world" {
		t.Fatalf("text=%q", got)
	}
	if seenPath != "/responses" {
		t.Fatalf("path=%q", seenPath)
	}
	if seenAuth != "Bearer sk-test" {
		t.Fatalf("auth=%q", seenAuth)
	}
	if seenAccount != "" {
		t.Fatalf("generic responses must not send chatgpt-account-id, got %q", seenAccount)
	}
	if seenModel != "gpt-5.4-mini" || !seenStream {
		t.Fatalf("model=%q stream=%v", seenModel, seenStream)
	}
	if !strings.Contains(seenPrompt, "You are helpful.") {
		t.Fatalf("instructions=%q", seenPrompt)
	}
	if seenCache != "sess-cache-key" {
		t.Fatalf("prompt_cache_key=%q want sess-cache-key", seenCache)
	}
	if len(seenInput) != 1 {
		t.Fatalf("input items=%d", len(seenInput))
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts=%d", len(attempts))
	}
	a := attempts[0]
	if a.Provider != "openai" || a.EndpointKind != providercontract.EndpointKindResponses {
		t.Fatalf("attempt provider/kind=%+v", a)
	}
	if a.StatusCode != http.StatusOK || a.Usage.TotalTokens != 17 || a.Usage.Cache.ReadTokens != 4 || !a.Usage.Cache.Hit {
		t.Fatalf("attempt=%+v", a)
	}
}

func TestResponsesModelRequestScopedCacheKeyOverridesConfig(t *testing.T) {
	var seenCache string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			PromptCacheKey string `json:"prompt_cache_key"`
		}
		_ = json.Unmarshal(raw, &req)
		seenCache = req.PromptCacheKey
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(), PromptCacheKey: "client-default",
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}
	ctx := providercontract.WithPromptCacheKey(context.Background(), "per-request-key")
	if _, err := collectResponses(llm.GenerateContent(ctx, &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false)); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if seenCache != "per-request-key" {
		t.Fatalf("prompt_cache_key=%q want per-request-key", seenCache)
	}
}

func TestResponsesModelFunctionCallRoundTrip(t *testing.T) {
	var seenInput []any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Input []any `json:"input"`
			Tools []any `json:"tools"`
		}
		_ = json.Unmarshal(raw, &req)
		seenInput = req.Input

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_abc\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Weather in SF?", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  &genai.Schema{Type: genai.TypeObject},
				}},
			}},
		},
	}
	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	var fc *genai.FunctionCall
	for _, p := range responses[0].Content.Parts {
		if p.FunctionCall != nil {
			fc = p.FunctionCall
		}
	}
	if fc == nil {
		t.Fatal("expected a function call part")
	}
	if fc.Name != "get_weather" || fc.ID != "call_abc" {
		t.Fatalf("function call=%+v", fc)
	}
	if city, _ := fc.Args["city"].(string); city != "SF" {
		t.Fatalf("args=%+v", fc.Args)
	}
	if len(seenInput) != 1 {
		t.Fatalf("input items=%d", len(seenInput))
	}
}

func TestResponsesModelAttemptSinkRecordsHTTPFailure(t *testing.T) {
	var attempts []providercontract.ModelAttempt
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(),
		AttemptSink: providercontract.AttemptSinkFunc(func(a providercontract.ModelAttempt) {
			attempts = append(attempts, a)
		}),
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}
	_, err = collectResponses(llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false))
	if err == nil {
		t.Fatal("expected error on 429")
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts=%d", len(attempts))
	}
	a := attempts[0]
	if a.StatusCode != http.StatusTooManyRequests || a.FailureReason != providercontract.FailoverReasonRateLimit {
		t.Fatalf("attempt=%+v", a)
	}
	if a.EndpointKind != providercontract.EndpointKindResponses {
		t.Fatalf("endpointKind=%v", a.EndpointKind)
	}
}

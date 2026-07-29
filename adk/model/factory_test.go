package model

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	providercontract "github.com/cpunion/ailib/adk/model/provider"
	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestParseModelString(t *testing.T) {
	tests := []struct {
		input        string
		wantProvider string
		wantModel    string
	}{
		{input: "openrouter:openai/gpt-5.1-codex", wantProvider: ProviderOpenRouter, wantModel: "openai/gpt-5.1-codex"},
		{input: "openai:gpt-5.1", wantProvider: ProviderOpenAI, wantModel: "gpt-5.1"},
		{input: "deepseek:deepseek-v4-flash", wantProvider: ProviderDeepSeek, wantModel: "deepseek-v4-flash"},
		{input: "minimax:MiniMax-M2.7", wantProvider: ProviderMiniMax, wantModel: "MiniMax-M2.7"},
		{input: "groq:llama-3.3-70b-versatile", wantProvider: ProviderGroq, wantModel: "llama-3.3-70b-versatile"},
		{input: "codex:gpt-5.4-mini", wantProvider: ProviderCodex, wantModel: "gpt-5.4-mini"},
		{input: "gemini:gemini-2.5-flash", wantProvider: ProviderGemini, wantModel: "gemini-2.5-flash"},
		{input: "mock:echo", wantProvider: ProviderMock, wantModel: "echo"},
		{input: "mock-echo:{\"repeat\":2}", wantProvider: ProviderMockEcho, wantModel: "{\"repeat\":2}"},
		{input: "openai/gpt-5.1-codex", wantProvider: ProviderOpenRouter, wantModel: "openai/gpt-5.1-codex"},
		{input: "CoDeX:gpt-5.4-mini", wantProvider: ProviderCodex, wantModel: "gpt-5.4-mini"},
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			gotProvider, gotModel := ParseModelString(tt.input)
			if gotProvider != tt.wantProvider || gotModel != tt.wantModel {
				t.Fatalf("ParseModelString(%q) = (%q, %q), want (%q, %q)", tt.input, gotProvider, gotModel, tt.wantProvider, tt.wantModel)
			}
		})
	}
}

func TestGetEnvHelpers(t *testing.T) {
	if got := GetAPIKeyEnvVar(ProviderCodex); got != "CODEX_API_KEY" {
		t.Fatalf("GetAPIKeyEnvVar(codex) = %q", got)
	}
	if got := GetBaseURLEnvVar(ProviderCodex); got != "CODEX_BASE_URL" {
		t.Fatalf("GetBaseURLEnvVar(codex) = %q", got)
	}
	if got := GetAPIKeyEnvVar(ProviderDeepSeek); got != "DEEPSEEK_API_KEY" {
		t.Fatalf("GetAPIKeyEnvVar(deepseek) = %q", got)
	}
	if got := GetBaseURLEnvVar(ProviderDeepSeek); got != "DEEPSEEK_BASE_URL" {
		t.Fatalf("GetBaseURLEnvVar(deepseek) = %q", got)
	}
	if got := GetDefaultBaseURL(ProviderDeepSeek); got != "https://api.deepseek.com/v1" {
		t.Fatalf("GetDefaultBaseURL(deepseek) = %q", got)
	}
	if got := GetAPIKeyEnvVar(ProviderMiniMax); got != "MINIMAX_API_KEY" {
		t.Fatalf("GetAPIKeyEnvVar(minimax) = %q", got)
	}
	if got := GetBaseURLEnvVar(ProviderMiniMax); got != "MINIMAX_BASE_URL" {
		t.Fatalf("GetBaseURLEnvVar(minimax) = %q", got)
	}
	if got := GetDefaultBaseURL(ProviderMiniMax); got != "https://api.minimax.io/v1" {
		t.Fatalf("GetDefaultBaseURL(minimax) = %q", got)
	}
	if got := GetAPIKeyEnvVar(ProviderGroq); got != "GROQ_API_KEY" {
		t.Fatalf("GetAPIKeyEnvVar(groq) = %q", got)
	}
	if got := GetBaseURLEnvVar(ProviderGroq); got != "GROQ_BASE_URL" {
		t.Fatalf("GetBaseURLEnvVar(groq) = %q", got)
	}
	if got := GetDefaultBaseURL(ProviderGroq); got != "https://api.groq.com/openai/v1" {
		t.Fatalf("GetDefaultBaseURL(groq) = %q", got)
	}
}

func TestNewFromConfigOpenAICompatibleAcceptsAttemptSink(t *testing.T) {
	var attempts []providercontract.ModelAttempt
	llm, err := NewFromConfig(context.Background(), Config{
		Provider:   ProviderOpenAI,
		Model:      "gpt-test",
		APIKey:     "sk-test",
		BaseURL:    "http://example.invalid/v1",
		HTTPClient: http.DefaultClient,
		AttemptSink: providercontract.AttemptSinkFunc(func(attempt providercontract.ModelAttempt) {
			attempts = append(attempts, attempt)
		}),
	})
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	if llm == nil {
		t.Fatal("expected llm")
	}
	if len(attempts) != 0 {
		t.Fatalf("attempts should only be observed during generation, got %d", len(attempts))
	}
}

func TestNewFromConfigOpenAICompatibleForwardsPromptCacheKey(t *testing.T) {
	var gotBody map[string]any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := json.NewDecoder(r.Body).Decode(&gotBody); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-cache","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer ts.Close()

	llm, err := NewFromConfig(context.Background(), Config{
		Provider:       ProviderOpenAI,
		Model:          "gpt-test",
		APIKey:         "sk-test",
		BaseURL:        ts.URL,
		HTTPClient:     ts.Client(),
		PromptCacheKey: "aos:tool-manifest:def456",
	})
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	for _, err := range llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if gotBody["prompt_cache_key"] != "aos:tool-manifest:def456" {
		t.Fatalf("prompt_cache_key=%v body=%+v", gotBody["prompt_cache_key"], gotBody)
	}
}

func TestNewFromConfigEndpointKindRoutesToResponses(t *testing.T) {
	var path string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewFromConfig(context.Background(), Config{
		Provider:     ProviderOpenAI,
		Model:        "gpt-test",
		APIKey:       "sk-test",
		BaseURL:      ts.URL,
		HTTPClient:   ts.Client(),
		EndpointKind: providercontract.EndpointKindResponses,
	})
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	for _, err := range llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if path != "/responses" {
		t.Fatalf("EndpointKindResponses must hit /responses, got %q", path)
	}
}

func TestNewFromConfigDefaultRoutesToChatCompletions(t *testing.T) {
	var path string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	defer ts.Close()

	llm, err := NewFromConfig(context.Background(), Config{
		Provider:   ProviderOpenAI,
		Model:      "gpt-test",
		APIKey:     "sk-test",
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	for _, err := range llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if path != "/chat/completions" {
		t.Fatalf("default must hit /chat/completions, got %q", path)
	}
}

func TestNewFromConfigCodexPrefersExplicitAPIKeyAccountID(t *testing.T) {
	const explicitToken = "eyJhbGciOiJub25lIn0.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF9leHBsaWNpdCJ9fQ."
	const envToken = "eyJhbGciOiJub25lIn0.eyJodHRwczovL2FwaS5vcGVuYWkuY29tL2F1dGgiOnsiY2hhdGdwdF9hY2NvdW50X2lkIjoiYWNjdF9lbnYifX0."

	var (
		seenAccount string
		seenEffort  string
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenAccount = r.Header.Get("chatgpt-account-id")
		var request struct {
			Reasoning struct {
				Effort string `json:"effort"`
			} `json:"reasoning"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		seenEffort = request.Reasoning.Effort
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	t.Setenv("CODEX_API_KEY", envToken)
	llm, err := NewFromConfig(context.Background(), Config{
		Provider:        ProviderCodex,
		Model:           "gpt-5.4-mini",
		APIKey:          explicitToken,
		BaseURL:         ts.URL + "/backend-api/codex",
		HTTPClient:      ts.Client(),
		ReasoningEffort: "xhigh",
	})
	if err != nil {
		t.Fatalf("NewFromConfig: %v", err)
	}
	for _, err := range llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
	}
	if seenAccount != "acct_explicit" {
		t.Fatalf("account=%q", seenAccount)
	}
	if seenEffort != "xhigh" {
		t.Fatalf("reasoning effort=%q", seenEffort)
	}
}

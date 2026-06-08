package model

import (
	"context"
	"net/http"
	"testing"

	providercontract "github.com/cpunion/ailib/adk/model/provider"
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

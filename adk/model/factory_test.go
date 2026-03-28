package model

import "testing"

func TestParseModelString(t *testing.T) {
	tests := []struct {
		input        string
		wantProvider string
		wantModel    string
	}{
		{input: "openrouter:openai/gpt-5.1-codex", wantProvider: ProviderOpenRouter, wantModel: "openai/gpt-5.1-codex"},
		{input: "openai:gpt-5.1", wantProvider: ProviderOpenAI, wantModel: "gpt-5.1"},
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
}

// Package model provides LLM model factory and utilities.
package model

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/genai"

	"github.com/cpunion/ailib/adk/model/codexauth"
	"github.com/cpunion/ailib/adk/model/openai"
)

// Provider constants.
const (
	ProviderOpenRouter = "openrouter"
	ProviderOpenAI     = "openai"
	ProviderCodex      = "codex"
	ProviderGemini     = "gemini"
	ProviderVertexAI   = "vertexai"
	ProviderMock       = "mock"
	ProviderMockEcho   = "mock-echo"
)

// Config holds model factory configuration.
type Config struct {
	Provider string
	Model    string
	APIKey   string
	BaseURL  string
}

// ParseModelString parses a model string with an optional provider prefix.
func ParseModelString(modelWithProvider string) (provider, model string) {
	raw := strings.TrimSpace(modelWithProvider)
	if raw == "" {
		return ProviderOpenRouter, ""
	}

	for _, providerName := range []string{
		ProviderMockEcho,
		ProviderOpenRouter,
		ProviderOpenAI,
		ProviderCodex,
		ProviderGemini,
		ProviderVertexAI,
		ProviderMock,
	} {
		if strings.EqualFold(raw, providerName) {
			return providerName, ""
		}
		prefix := providerName + ":"
		if len(raw) >= len(prefix) && strings.EqualFold(raw[:len(prefix)], prefix) {
			return providerName, raw[len(prefix):]
		}
	}
	return ProviderOpenRouter, raw
}

// GetAPIKeyEnvVar returns the environment variable name for the provider API key.
func GetAPIKeyEnvVar(provider string) string {
	switch provider {
	case ProviderOpenRouter:
		return "OPENROUTER_API_KEY"
	case ProviderOpenAI:
		return "OPENAI_API_KEY"
	case ProviderCodex:
		return "CODEX_API_KEY"
	case ProviderGemini:
		return "GEMINI_API_KEY"
	case ProviderVertexAI:
		return ""
	default:
		return ""
	}
}

// GetBaseURLEnvVar returns the environment variable name for the provider base URL override.
func GetBaseURLEnvVar(provider string) string {
	switch provider {
	case ProviderOpenRouter:
		return "OPENROUTER_BASE_URL"
	case ProviderOpenAI:
		return "OPENAI_BASE_URL"
	case ProviderCodex:
		return "CODEX_BASE_URL"
	default:
		return ""
	}
}

// GetDefaultBaseURL returns the default base URL for the provider.
func GetDefaultBaseURL(provider string) string {
	switch provider {
	case ProviderOpenRouter:
		return "https://openrouter.ai/api/v1"
	case ProviderOpenAI:
		return "https://api.openai.com/v1"
	case ProviderCodex:
		return codexauth.OpenAIBaseURL
	case ProviderGemini:
		return ""
	case ProviderVertexAI:
		return ""
	default:
		return ""
	}
}

// New creates an LLM model based on the provider prefix in the model string.
func New(ctx context.Context, modelWithProvider string) (adkmodel.LLM, error) {
	return NewWith(ctx, modelWithProvider, "", "")
}

// NewWith creates an LLM model based on the provider prefix in the model string.
func NewWith(ctx context.Context, modelWithProvider, apiKey, baseURL string) (adkmodel.LLM, error) {
	provider, modelName := ParseModelString(modelWithProvider)

	if apiKey == "" && provider != ProviderCodex {
		envVar := GetAPIKeyEnvVar(provider)
		if envVar != "" {
			apiKey = os.Getenv(envVar)
		}
	}

	if baseURL == "" && provider != ProviderCodex {
		if envVar := GetBaseURLEnvVar(provider); envVar != "" {
			if v := strings.TrimSpace(os.Getenv(envVar)); v != "" {
				baseURL = v
			}
		}
	}

	if baseURL == "" && provider != ProviderCodex {
		baseURL = GetDefaultBaseURL(provider)
	}

	var (
		llm adkmodel.LLM
		err error
	)
	switch provider {
	case ProviderOpenRouter, ProviderOpenAI:
		if apiKey == "" {
			return nil, fmt.Errorf("API key required for %s provider (set %s)", provider, GetAPIKeyEnvVar(provider))
		}
		llm, err = openai.NewModel(ctx, modelName, &openai.ClientConfig{
			APIKey:   apiKey,
			BaseURL:  baseURL,
			Provider: provider,
		})
		if err != nil {
			return nil, err
		}

	case ProviderCodex:
		resolution := codexauth.Resolve(codexauth.ResolveOptions{})
		if strings.TrimSpace(apiKey) == "" {
			apiKey = strings.TrimSpace(resolution.APIKey)
		}
		accountID := strings.TrimSpace(resolution.AccountID)
		if accountID == "" {
			accountID = strings.TrimSpace(codexauth.ExtractAccountID(apiKey))
		}
		if strings.TrimSpace(apiKey) == "" {
			return nil, fmt.Errorf("API key required for %s provider (checked CODEX_API_KEY, macOS Keychain, $CODEX_HOME/auth.json, then OPENAI_API_KEY)", provider)
		}
		if strings.TrimSpace(baseURL) == "" {
			baseURL = codexauth.ResolveBaseURL(resolution.Source, nil)
		}
		if accountID != "" && strings.Contains(strings.TrimSpace(baseURL), "/backend-api/codex") {
			llm, err = openai.NewCodexResponsesModel(ctx, modelName, &openai.ClientConfig{
				APIKey:   apiKey,
				BaseURL:  baseURL,
				Provider: provider,
			}, accountID)
		} else {
			llm, err = openai.NewModel(ctx, modelName, &openai.ClientConfig{
				APIKey:   apiKey,
				BaseURL:  baseURL,
				Provider: provider,
			})
		}
		if err != nil {
			return nil, err
		}

	case ProviderGemini:
		cfg := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
		if apiKey != "" {
			cfg.APIKey = apiKey
		}
		llm, err = gemini.NewModel(ctx, modelName, cfg)
		if err != nil {
			return nil, err
		}

	case ProviderVertexAI:
		cfg := &genai.ClientConfig{Backend: genai.BackendVertexAI}
		if apiKey != "" {
			cfg.APIKey = apiKey
		}
		llm, err = gemini.NewModel(ctx, modelName, cfg)
		if err != nil {
			return nil, err
		}

	case ProviderMockEcho:
		cfg, err := ParseMockEchoConfig(modelName)
		if err != nil {
			return nil, err
		}
		llm = NewMockEchoLLMWithConfig(cfg)

	case ProviderMock:
		specModelName := strings.TrimSpace(modelName)
		if strings.EqualFold(specModelName, "echo") || strings.EqualFold(specModelName, "echo:") {
			llm = NewMockEchoLLM()
			break
		}
		echoPrefix := "echo:"
		if len(specModelName) >= len(echoPrefix) && strings.EqualFold(specModelName[:len(echoPrefix)], echoPrefix) {
			cfg, err := ParseMockEchoConfig(specModelName[len(echoPrefix):])
			if err != nil {
				return nil, err
			}
			llm = NewMockEchoLLMWithConfig(cfg)
			break
		}
		if specRaw, err := ResolveMockSpec(specModelName); err != nil {
			return nil, err
		} else if strings.HasPrefix(strings.TrimSpace(specRaw), "{") {
			var spec MockSpec
			if err := json.Unmarshal([]byte(specRaw), &spec); err != nil {
				return nil, fmt.Errorf("invalid mock spec: %w", err)
			}
			llm = NewMockSpecLLM(spec)
			break
		}
		text := strings.TrimSpace(specModelName)
		if text == "" {
			text = "MOCK_OK"
		}
		llm = NewMockLLM(&adkmodel.LLMResponse{
			Content:      genai.NewContentFromText(text, genai.RoleModel),
			Partial:      false,
			TurnComplete: true,
		})

	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}

	return llm, nil
}

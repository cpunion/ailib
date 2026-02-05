// Package model provides LLM model factory and utilities.
package model

import (
	"context"
	"fmt"
	"os"
	"strings"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/adk/model/gemini"
	"google.golang.org/genai"

	"github.com/cpunion/ailib/adk/model/openai"
)

// Provider constants
const (
	ProviderOpenRouter = "openrouter"
	ProviderOpenAI     = "openai"
	ProviderGemini     = "gemini"
	ProviderVertexAI   = "vertexai"
	ProviderMock       = "mock"
)

// Config holds model factory configuration
type Config struct {
	// Provider is the LLM provider (openrouter, openai, gemini, mock)
	Provider string
	// Model is the model name without provider prefix
	Model string
	// APIKey is the API key for the provider
	APIKey string
	// BaseURL is the API base URL (optional, uses default if empty)
	BaseURL string
}

// ParseModelString parses a model string with optional provider prefix.
// Format: provider:model or just model (defaults to openrouter)
// Examples:
//   - "openrouter:openai/gpt-5.1-codex" → provider=openrouter, model=openai/gpt-5.1-codex
//   - "openai:gpt-5.1-codex" → provider=openai, model=gpt-5.1-codex
//   - "gemini:gemini-3-pro" → provider=gemini, model=gemini-3-pro
//   - "vertexai:gemini-2.5-flash" → provider=vertexai, model=gemini-2.5-flash
//   - "mock:" → provider=mock, model=""
//   - "openai/gpt-5.1-codex" → provider=openrouter, model=openai/gpt-5.1-codex (legacy)
func ParseModelString(modelWithProvider string) (provider, model string) {
	if strings.HasPrefix(modelWithProvider, ProviderOpenRouter+":") {
		return ProviderOpenRouter, strings.TrimPrefix(modelWithProvider, ProviderOpenRouter+":")
	}
	if strings.HasPrefix(modelWithProvider, ProviderOpenAI+":") {
		return ProviderOpenAI, strings.TrimPrefix(modelWithProvider, ProviderOpenAI+":")
	}
	if strings.HasPrefix(modelWithProvider, ProviderGemini+":") {
		return ProviderGemini, strings.TrimPrefix(modelWithProvider, ProviderGemini+":")
	}
	if strings.HasPrefix(modelWithProvider, ProviderVertexAI+":") {
		return ProviderVertexAI, strings.TrimPrefix(modelWithProvider, ProviderVertexAI+":")
	}
	if strings.HasPrefix(modelWithProvider, ProviderMock+":") {
		return ProviderMock, strings.TrimPrefix(modelWithProvider, ProviderMock+":")
	}
	// Default to openrouter for backward compatibility
	return ProviderOpenRouter, modelWithProvider
}

// GetAPIKeyEnvVar returns the environment variable name for the provider's API key.
func GetAPIKeyEnvVar(provider string) string {
	switch provider {
	case ProviderOpenRouter:
		return "OPENROUTER_API_KEY"
	case ProviderOpenAI:
		return "OPENAI_API_KEY"
	case ProviderGemini:
		return "GEMINI_API_KEY"
	case ProviderVertexAI:
		return ""
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
	case ProviderGemini:
		return "" // Gemini uses SDK default
	case ProviderVertexAI:
		return "" // Vertex AI uses SDK default
	default:
		return ""
	}
}

// New creates an LLM model based on the provider prefix in the model string.
// API keys and base URL overrides are loaded from environment variables.
func New(ctx context.Context, modelWithProvider string) (adkmodel.LLM, error) {
	return NewWith(ctx, modelWithProvider, "", "")
}

// NewWith creates an LLM model based on the provider prefix in the model string.
// The apiKey and baseURL parameters are optional overrides; if empty, they are
// loaded from environment variables based on the provider.
func NewWith(ctx context.Context, modelWithProvider, apiKey, baseURL string) (adkmodel.LLM, error) {
	provider, modelName := ParseModelString(modelWithProvider)

	// Get API key from environment if not provided
	if apiKey == "" {
		envVar := GetAPIKeyEnvVar(provider)
		if envVar != "" {
			apiKey = os.Getenv(envVar)
		}
	}

	// Get default base URL if not provided
	if baseURL == "" {
		baseURL = GetDefaultBaseURL(provider)
	}

	switch provider {
	case ProviderOpenRouter, ProviderOpenAI:
		// Both use OpenAI-compatible API
		if apiKey == "" {
			return nil, fmt.Errorf("API key required for %s provider (set %s)", provider, GetAPIKeyEnvVar(provider))
		}
		return openai.NewModel(ctx, modelName, &openai.ClientConfig{
			APIKey:  apiKey,
			BaseURL: baseURL,
		})

	case ProviderGemini:
		// Use ADK's native Gemini support. Let the SDK pick up env/config if API key is empty.
		cfg := &genai.ClientConfig{Backend: genai.BackendGeminiAPI}
		if apiKey != "" {
			cfg.APIKey = apiKey
		}
		// Note: Gemini SDK doesn't support custom BaseURL
		return gemini.NewModel(ctx, modelName, cfg)

	case ProviderVertexAI:
		// Use Vertex AI backend (ADC + GOOGLE_CLOUD_PROJECT/LOCATION).
		cfg := &genai.ClientConfig{Backend: genai.BackendVertexAI}
		if apiKey != "" {
			cfg.APIKey = apiKey
		}
		return gemini.NewModel(ctx, modelName, cfg)

	case ProviderMock:
		// Return empty mock LLM (for testing)
		return NewMockLLM(), nil

	default:
		return nil, fmt.Errorf("unknown provider: %s", provider)
	}
}

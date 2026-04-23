package model

import (
	"context"
	"fmt"
	"strings"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

// MultimodalOptions overrides provider credentials for a single call.
type MultimodalOptions struct {
	APIKey  string
	BaseURL string
}

// MultimodalResult is the normalized single-turn result for text/image calls.
type MultimodalResult struct {
	Response *adkmodel.LLMResponse
	Text     string
	Images   []*genai.Blob
}

// GenerateOnce creates a provider-backed model and returns the final response.
func GenerateOnce(ctx context.Context, modelWithProvider string, req *adkmodel.LLMRequest, opts *MultimodalOptions) (*MultimodalResult, error) {
	var apiKey, baseURL string
	if opts != nil {
		apiKey = strings.TrimSpace(opts.APIKey)
		baseURL = strings.TrimSpace(opts.BaseURL)
	}
	llm, err := NewWith(ctx, modelWithProvider, apiKey, baseURL)
	if err != nil {
		return nil, err
	}
	return GenerateOnceWithLLM(ctx, llm, req)
}

// GenerateOnceWithLLM returns the final response from an existing LLM instance.
func GenerateOnceWithLLM(ctx context.Context, llm adkmodel.LLM, req *adkmodel.LLMRequest) (*MultimodalResult, error) {
	if llm == nil {
		return nil, fmt.Errorf("llm is nil")
	}
	if req == nil {
		req = &adkmodel.LLMRequest{}
	}

	var finalResp *adkmodel.LLMResponse
	for resp, err := range llm.GenerateContent(ctx, req, false) {
		if err != nil {
			return nil, err
		}
		finalResp = resp
	}
	if finalResp == nil {
		return nil, fmt.Errorf("model returned no response")
	}

	return &MultimodalResult{
		Response: finalResp,
		Text:     ExtractText(finalResp.Content),
		Images:   ExtractInlineData(finalResp.Content),
	}, nil
}

// NewTextRequest builds a single-turn text request.
func NewTextRequest(prompt string, config *genai.GenerateContentConfig) *adkmodel.LLMRequest {
	return &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText(prompt, genai.RoleUser),
		},
		Config: config,
	}
}

// NewImageRequest builds a single-turn image request and ensures IMAGE modality is enabled.
func NewImageRequest(prompt string, config *genai.GenerateContentConfig, references ...*genai.Blob) *adkmodel.LLMRequest {
	cfg := cloneGenerateContentConfig(config)
	if !hasResponseModality(cfg, "IMAGE") {
		cfg.ResponseModalities = append(cfg.ResponseModalities, "IMAGE")
	}

	parts := make([]*genai.Part, 0, 1+len(references))
	if strings.TrimSpace(prompt) != "" {
		parts = append(parts, genai.NewPartFromText(prompt))
	}
	for _, reference := range references {
		if reference == nil || len(reference.Data) == 0 {
			continue
		}
		parts = append(parts, &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: reference.MIMEType,
				Data:     append([]byte(nil), reference.Data...),
			},
		})
	}

	return &adkmodel.LLMRequest{
		Contents: []*genai.Content{{
			Role:  genai.RoleUser,
			Parts: parts,
		}},
		Config: cfg,
	}
}

// ExtractText joins all non-thought text parts from a content payload.
func ExtractText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	textParts := make([]string, 0, len(content.Parts))
	for _, part := range content.Parts {
		if part == nil || part.Thought || strings.TrimSpace(part.Text) == "" {
			continue
		}
		textParts = append(textParts, strings.TrimSpace(part.Text))
	}
	return strings.Join(textParts, "\n")
}

// ExtractInlineData collects binary parts from a content payload.
func ExtractInlineData(content *genai.Content) []*genai.Blob {
	if content == nil {
		return nil
	}
	images := make([]*genai.Blob, 0, len(content.Parts))
	for _, part := range content.Parts {
		if part == nil || part.InlineData == nil || len(part.InlineData.Data) == 0 {
			continue
		}
		images = append(images, &genai.Blob{
			MIMEType: part.InlineData.MIMEType,
			Data:     append([]byte(nil), part.InlineData.Data...),
		})
	}
	return images
}

func cloneGenerateContentConfig(config *genai.GenerateContentConfig) *genai.GenerateContentConfig {
	if config == nil {
		return &genai.GenerateContentConfig{}
	}
	cloned := *config
	if len(config.ResponseModalities) > 0 {
		cloned.ResponseModalities = append([]string(nil), config.ResponseModalities...)
	}
	if len(config.StopSequences) > 0 {
		cloned.StopSequences = append([]string(nil), config.StopSequences...)
	}
	if len(config.SafetySettings) > 0 {
		cloned.SafetySettings = append([]*genai.SafetySetting(nil), config.SafetySettings...)
	}
	if len(config.Tools) > 0 {
		cloned.Tools = append([]*genai.Tool(nil), config.Tools...)
	}
	if len(config.Labels) > 0 {
		cloned.Labels = make(map[string]string, len(config.Labels))
		for key, value := range config.Labels {
			cloned.Labels[key] = value
		}
	}
	return &cloned
}

func hasResponseModality(config *genai.GenerateContentConfig, target string) bool {
	if config == nil {
		return false
	}
	for _, modality := range config.ResponseModalities {
		if strings.EqualFold(strings.TrimSpace(modality), target) {
			return true
		}
	}
	return false
}

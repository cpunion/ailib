package model

import (
	"context"
	"testing"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestGenerateOnceWithLLMExtractsTextAndImages(t *testing.T) {
	llm := NewMockLLM(&adkmodel.LLMResponse{
		Content: &genai.Content{
			Role: genai.RoleModel,
			Parts: []*genai.Part{
				{Text: "hello"},
				{Thought: true, Text: "hidden"},
				{InlineData: &genai.Blob{MIMEType: "image/png", Data: []byte("img-1")}},
			},
		},
	})

	got, err := GenerateOnceWithLLM(context.Background(), llm, NewTextRequest("say hello", nil))
	if err != nil {
		t.Fatalf("GenerateOnceWithLLM() error = %v", err)
	}
	if got.Text != "hello" {
		t.Fatalf("text = %q, want hello", got.Text)
	}
	if len(got.Images) != 1 {
		t.Fatalf("images = %d, want 1", len(got.Images))
	}
	if got.Images[0].MIMEType != "image/png" || string(got.Images[0].Data) != "img-1" {
		t.Fatalf("image = %+v", got.Images[0])
	}
}

func TestNewImageRequestAddsImageModalityAndCopiesReferences(t *testing.T) {
	reference := &genai.Blob{MIMEType: "image/jpeg", Data: []byte("ref")}
	cfg := &genai.GenerateContentConfig{
		ResponseModalities: []string{"TEXT"},
		ImageConfig:        &genai.ImageConfig{AspectRatio: "16:9"},
	}

	req := NewImageRequest("draw a cover", cfg, reference)
	if req == nil || len(req.Contents) != 1 {
		t.Fatalf("request contents = %+v", req)
	}
	if len(req.Contents[0].Parts) != 2 {
		t.Fatalf("parts = %d, want 2", len(req.Contents[0].Parts))
	}
	if req.Config == nil || !hasResponseModality(req.Config, "IMAGE") || !hasResponseModality(req.Config, "TEXT") {
		t.Fatalf("modalities = %+v", req.Config)
	}
	if !hasResponseModality(cfg, "TEXT") || hasResponseModality(cfg, "IMAGE") {
		t.Fatalf("original config should stay unchanged: %+v", cfg.ResponseModalities)
	}
	req.Contents[0].Parts[1].InlineData.Data[0] = 'X'
	if string(reference.Data) != "ref" {
		t.Fatalf("reference blob should not be mutated: %q", string(reference.Data))
	}
}

func TestExtractTextJoinsNonThoughtParts(t *testing.T) {
	content := &genai.Content{
		Parts: []*genai.Part{
			{Text: "alpha"},
			{Thought: true, Text: "skip"},
			{Text: "beta"},
		},
	}
	if got := ExtractText(content); got != "alpha\nbeta" {
		t.Fatalf("ExtractText() = %q, want alpha\\nbeta", got)
	}
}

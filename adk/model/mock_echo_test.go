package model

import (
	"context"
	"testing"
	"time"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestMockEchoLLMGenerateContentUsesLatestUserText(t *testing.T) {
	llm := NewMockEchoLLMWithConfig(MockEchoConfig{Repeat: 1, Delay: 0, Prefix: "echo: "})
	req := &adkmodel.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("first", genai.RoleUser),
			genai.NewContentFromText("hello", genai.RoleUser),
		},
	}

	var got *adkmodel.LLMResponse
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent err: %v", err)
		}
		got = resp
	}
	if got == nil || got.Content == nil || len(got.Content.Parts) == 0 {
		t.Fatalf("empty response: %+v", got)
	}
	if got.Content.Parts[0].Text != "echo: hello" {
		t.Fatalf("text=%q", got.Content.Parts[0].Text)
	}
}

func TestMockEchoLLMStreamByRune(t *testing.T) {
	llm := NewMockEchoLLMWithConfig(MockEchoConfig{Repeat: 1, Delay: 0, Prefix: "echo: "})
	req := &adkmodel.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)}}

	var partials []string
	var final string
	for resp, err := range llm.GenerateContent(context.Background(), req, true) {
		if err != nil {
			t.Fatalf("GenerateContent err: %v", err)
		}
		if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 {
			continue
		}
		text := resp.Content.Parts[0].Text
		if resp.Partial {
			partials = append(partials, text)
			continue
		}
		final = text
	}
	if len(partials) == 0 {
		t.Fatalf("expected partial responses")
	}
	if final != "echo: hello" {
		t.Fatalf("final=%q", final)
	}
	if partials[len(partials)-1] != final {
		t.Fatalf("last partial=%q final=%q", partials[len(partials)-1], final)
	}
}

func TestParseMockEchoConfig(t *testing.T) {
	cfg, err := ParseMockEchoConfig("")
	if err != nil {
		t.Fatalf("ParseMockEchoConfig default: %v", err)
	}
	if cfg.Repeat != 1 || cfg.Delay != 100*time.Millisecond || cfg.Prefix != "echo: " {
		t.Fatalf("cfg=%+v", cfg)
	}

	cfg, err = ParseMockEchoConfig(`{"repeat":3,"delay":"1s","prefix":"demo: "}`)
	if err != nil {
		t.Fatalf("ParseMockEchoConfig json: %v", err)
	}
	if cfg.Repeat != 3 || cfg.Delay != time.Second || cfg.Prefix != "demo: " {
		t.Fatalf("cfg=%+v", cfg)
	}
}

func TestParseMockEchoModel(t *testing.T) {
	cfg, err := ParseMockEchoModel(`mock-echo:{"repeat":2,"delay":"2s","prefix":"x: "}`)
	if err != nil {
		t.Fatalf("ParseMockEchoModel: %v", err)
	}
	if cfg.Repeat != 2 || cfg.Delay != 2*time.Second || cfg.Prefix != "x: " {
		t.Fatalf("cfg=%+v", cfg)
	}
}

func TestFactoryMockVariants(t *testing.T) {
	llm, err := NewWith(context.Background(), `mock-echo:{"repeat":2,"delay":"0s","prefix":"echo: "}`, "", "")
	if err != nil {
		t.Fatalf("NewWith mock-echo: %v", err)
	}
	if llm.Name() != "mock-echo" {
		t.Fatalf("name=%q", llm.Name())
	}

	req := &adkmodel.LLMRequest{Contents: []*genai.Content{genai.NewContentFromText("hello", genai.RoleUser)}}
	var got string
	for resp, err := range llm.GenerateContent(context.Background(), req, false) {
		if err != nil {
			t.Fatalf("GenerateContent: %v", err)
		}
		if resp != nil && resp.Content != nil && len(resp.Content.Parts) > 0 {
			got = resp.Content.Parts[0].Text
		}
	}
	if got != "echo: hello\necho: hello" {
		t.Fatalf("text=%q", got)
	}

	llm, err = NewWith(context.Background(), "mock:echo", "", "")
	if err != nil {
		t.Fatalf("NewWith mock:echo: %v", err)
	}
	if llm.Name() != "mock-echo" {
		t.Fatalf("name=%q", llm.Name())
	}

	llm, err = NewWith(context.Background(), "mock:hello", "", "")
	if err != nil {
		t.Fatalf("NewWith mock literal: %v", err)
	}
	if llm.Name() != "mock-llm" {
		t.Fatalf("name=%q", llm.Name())
	}
}

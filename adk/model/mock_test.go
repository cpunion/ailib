package model

import (
	"context"
	"testing"

	"iter"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestMockLLMSingleTurn(t *testing.T) {
	mock := NewMockLLM(
		&adkmodel.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel), Partial: true},
		&adkmodel.LLMResponse{Content: genai.NewContentFromText("hello world", genai.RoleModel), Partial: false, TurnComplete: true},
	)

	ctx := context.Background()
	resp, err := readAll(mock.GenerateContent(ctx, &adkmodel.LLMRequest{Model: "test"}, true))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp) != 2 {
		t.Fatalf("expected 2 responses, got %d", len(resp))
	}
	if got := resp[0].Content.Parts[0].Text; got != "hello" {
		t.Fatalf("first response = %q, want hello", got)
	}
	if got := resp[1].Content.Parts[0].Text; got != "hello world" {
		t.Fatalf("second response = %q, want hello world", got)
	}
}

func TestMockLLMNonStreaming(t *testing.T) {
	mock := NewMockLLM(
		&adkmodel.LLMResponse{Content: genai.NewContentFromText("partial", genai.RoleModel), Partial: true},
		&adkmodel.LLMResponse{Content: genai.NewContentFromText("final", genai.RoleModel), Partial: false},
	)

	ctx := context.Background()
	// Non-streaming should only return the last (final) response
	resp, err := readAll(mock.GenerateContent(ctx, &adkmodel.LLMRequest{Model: "test"}, false))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp) != 1 {
		t.Fatalf("expected 1 response for non-streaming, got %d", len(resp))
	}
	if got := resp[0].Content.Parts[0].Text; got != "final" {
		t.Fatalf("response = %q, want final", got)
	}
}

func TestMockConversationMultiTurn(t *testing.T) {
	conv := NewMockConversation(
		&Turn{Responses: []*adkmodel.LLMResponse{
			{Content: genai.NewContentFromText("first", genai.RoleModel)},
		}},
		&Turn{Responses: []*adkmodel.LLMResponse{
			{Content: genai.NewContentFromText("second", genai.RoleModel)},
		}},
	)

	ctx := context.Background()
	first, err := readAll(conv.GenerateContent(ctx, &adkmodel.LLMRequest{Model: "test"}, false))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := first[0].Content.Parts[0].Text; got != "first" {
		t.Fatalf("first response = %q, want first", got)
	}

	second, err := readAll(conv.GenerateContent(ctx, &adkmodel.LLMRequest{Model: "test"}, false))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := second[0].Content.Parts[0].Text; got != "second" {
		t.Fatalf("second response = %q, want second", got)
	}
}

func TestMockLLMEmpty(t *testing.T) {
	mock := NewMockLLM()
	resp, err := readAll(mock.GenerateContent(context.Background(), &adkmodel.LLMRequest{}, false))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp) != 0 {
		t.Fatalf("expected 0 responses for empty mock, got %d", len(resp))
	}
}

func readAll(seq iter.Seq2[*adkmodel.LLMResponse, error]) ([]*adkmodel.LLMResponse, error) {
	var res []*adkmodel.LLMResponse
	for resp, err := range seq {
		if err != nil {
			return nil, err
		}
		res = append(res, resp)
	}
	return res, nil
}

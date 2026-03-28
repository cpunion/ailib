package model

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestMockSpecLLMStreamingAccumulates(t *testing.T) {
	llm := NewMockSpecLLM(MockSpec{
		Stream:       true,
		ChunkDelayMs: 0,
		Chunks:       []string{"a", "b"},
		Usage:        &MockUsage{PromptTokens: 1, CompletionTokens: 2, TotalTokens: 3},
	})

	resp, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{Model: "test"}, true))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(resp))
	}
	if got := textOf(resp[0]); got != "a" || !resp[0].Partial {
		t.Fatalf("first = %q partial=%v, want %q partial=true", got, resp[0].Partial, "a")
	}
	if got := textOf(resp[1]); got != "ab" || !resp[1].Partial {
		t.Fatalf("second = %q partial=%v, want %q partial=true", got, resp[1].Partial, "ab")
	}
	if got := textOf(resp[2]); got != "ab" || resp[2].Partial {
		t.Fatalf("final = %q partial=%v, want %q partial=false", got, resp[2].Partial, "ab")
	}
	if resp[2].UsageMetadata == nil || resp[2].UsageMetadata.TotalTokenCount != 3 {
		t.Fatalf("expected usage metadata on final response")
	}
}

func TestResolveMockSpecFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "spec.txt")
	if err := os.WriteFile(path, []byte("hello"), 0o600); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	got, err := ResolveMockSpec("@" + path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "hello" {
		t.Fatalf("got %q, want hello", got)
	}
}

func TestNewMockInlineJSONSpec(t *testing.T) {
	llm, err := New(context.Background(), `mock:{"stream":true,"chunkDelayMs":0,"chunks":["x","y"]}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{Model: "test"}, true))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(resp) != 3 {
		t.Fatalf("expected 3 responses, got %d", len(resp))
	}
	if got := textOf(resp[2]); got != "xy" {
		t.Fatalf("final = %q, want xy", got)
	}
}

func TestMockSpecLLMTurnsAdvanceAndEmitFunctionCalls(t *testing.T) {
	llm := NewMockSpecLLM(MockSpec{
		Turns: []MockTurnSpec{
			{
				Responses: []MockResponseSpec{
					{
						Parts: []MockPartSpec{
							{
								FunctionCall: &MockFunctionCallSpec{
									ID:   "tc-mock-1",
									Name: "exec",
									Args: map[string]any{"command": "echo hi"},
								},
							},
						},
						TurnComplete: true,
					},
				},
			},
			{
				Responses: []MockResponseSpec{
					{
						Text:         "tool finished",
						TurnComplete: true,
					},
				},
			},
		},
	})

	first, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{Model: "test"}, true))
	if err != nil {
		t.Fatalf("first turn error: %v", err)
	}
	if len(first) != 1 {
		t.Fatalf("first turn responses=%d, want 1", len(first))
	}
	if first[0].Content == nil || len(first[0].Content.Parts) != 1 || first[0].Content.Parts[0].FunctionCall == nil {
		t.Fatalf("first turn missing function call: %+v", first[0])
	}
	if got := first[0].Content.Parts[0].FunctionCall.Name; got != "exec" {
		t.Fatalf("function call name=%q", got)
	}

	second, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{Model: "test"}, true))
	if err != nil {
		t.Fatalf("second turn error: %v", err)
	}
	if len(second) != 1 {
		t.Fatalf("second turn responses=%d, want 1", len(second))
	}
	if got := textOf(second[0]); got != "tool finished" {
		t.Fatalf("second turn text=%q", got)
	}
}

func TestMockSpecLLMRequestMatching(t *testing.T) {
	llm := NewMockSpecLLM(MockSpec{
		Turns: []MockTurnSpec{
			{
				ExpectLatestUserText: "write note",
				Responses: []MockResponseSpec{
					{Text: "matched user turn", TurnComplete: true},
				},
			},
			{
				ExpectFunctionResponseName:   "write",
				ExpectFunctionResponseID:     "tc-write-1",
				ExpectFunctionResponseStatus: "done",
				Responses: []MockResponseSpec{
					{Text: "matched tool result turn", TurnComplete: true},
				},
			},
		},
	})

	first, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{
		Model:    "test",
		Contents: []*genai.Content{genai.NewContentFromText("please write note", genai.RoleUser)},
	}, true))
	if err != nil {
		t.Fatalf("first turn error: %v", err)
	}
	if len(first) != 1 || textOf(first[0]) != "matched user turn" {
		t.Fatalf("unexpected first turn=%v", first)
	}

	respContent := genai.NewContentFromFunctionResponse("write", map[string]any{"status": "done"}, genai.RoleUser)
	respContent.Parts[0].FunctionResponse.ID = "tc-write-1"
	second, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{
		Model:    "test",
		Contents: []*genai.Content{respContent},
	}, true))
	if err != nil {
		t.Fatalf("second turn error: %v", err)
	}
	if len(second) != 1 || textOf(second[0]) != "matched tool result turn" {
		t.Fatalf("unexpected second turn=%v", second)
	}
}

func textOf(resp *adkmodel.LLMResponse) string {
	if resp == nil || resp.Content == nil || len(resp.Content.Parts) == 0 || resp.Content.Parts[0] == nil {
		return ""
	}
	return resp.Content.Parts[0].Text
}

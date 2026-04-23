package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestCodexResponsesModelGenerateText(t *testing.T) {
	var (
		seenAuth      string
		seenAccountID string
		seenModel     string
		seenStore     bool
		seenStream    bool
		seenPrompt    string
		seenInput     []any
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/responses" {
			t.Fatalf("path=%q", r.URL.Path)
		}
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var req struct {
			Model        string `json:"model"`
			Store        bool   `json:"store"`
			Stream       bool   `json:"stream"`
			Instructions string `json:"instructions"`
			Input        []any  `json:"input"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		seenAuth = r.Header.Get("Authorization")
		seenAccountID = r.Header.Get("chatgpt-account-id")
		seenModel = req.Model
		seenStore = req.Store
		seenStream = req.Stream
		seenPrompt = req.Instructions
		seenInput = req.Input

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\" world\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\",\"usage\":{\"input_tokens\":12,\"output_tokens\":5,\"total_tokens\":17}}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey:     "jwt-token",
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
		Provider:   "codex",
	}, "acct_123")
	if err != nil {
		t.Fatalf("NewCodexResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Say hello.", genai.RoleUser),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("Runtime: repo=/workspace | thinking=xhigh", "system"),
		},
	}
	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	gotText := extractPartsText(responses[0].Content)
	if gotText != "Hello world" {
		t.Fatalf("text=%q", gotText)
	}
	if seenAuth != "Bearer jwt-token" {
		t.Fatalf("auth=%q", seenAuth)
	}
	if seenAccountID != "acct_123" {
		t.Fatalf("account=%q", seenAccountID)
	}
	if seenModel != "gpt-5.4-mini" {
		t.Fatalf("model=%q", seenModel)
	}
	if seenStore {
		t.Fatalf("store should be false")
	}
	if !seenStream {
		t.Fatalf("stream should be true")
	}
	if !strings.Contains(seenPrompt, "thinking=xhigh") {
		t.Fatalf("instructions=%q", seenPrompt)
	}
	if len(seenInput) != 1 {
		t.Fatalf("input items=%d", len(seenInput))
	}
}

func TestCodexResponsesModelFunctionCallRoundTrip(t *testing.T) {
	var seenTools []any
	var seenInput []any

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var req struct {
			Tools []any `json:"tools"`
			Input []any `json:"input"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		seenTools = req.Tools
		seenInput = req.Input

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_add\",\"name\":\"add\",\"arguments\":\"\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\"{\\\"a\\\":2\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.function_call_arguments.delta\",\"item_id\":\"fc_1\",\"delta\":\",\\\"b\\\":3}\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_add\",\"name\":\"add\",\"arguments\":\"{\\\"a\\\":2,\\\"b\\\":3}\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey:     "jwt-token",
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
		Provider:   "codex",
	}, "acct_123")
	if err != nil {
		t.Fatalf("NewCodexResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("What is 2+3?", genai.RoleUser),
			{
				Role: "model",
				Parts: []*genai.Part{
					func() *genai.Part {
						part := genai.NewPartFromFunctionCall("add", map[string]any{"a": 2, "b": 3})
						part.FunctionCall.ID = "call_add"
						return part
					}(),
				},
			},
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							ID:       "call_add",
							Name:     "add",
							Response: map[string]any{"result": 5},
						},
					},
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "add",
							Description: "Add two integers",
							ParametersJsonSchema: map[string]any{
								"type": "object",
								"properties": map[string]any{
									"a": map[string]any{"type": "integer"},
									"b": map[string]any{"type": "integer"},
									"__aos": map[string]any{
										"type": "string",
									},
								},
								"required": []string{"a", "b"},
							},
						},
					},
				},
			},
		},
	}

	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	var gotCall *genai.FunctionCall
	for _, part := range responses[0].Content.Parts {
		if part.FunctionCall != nil {
			gotCall = part.FunctionCall
			break
		}
	}
	if gotCall == nil {
		t.Fatal("expected function call")
	}
	if gotCall.ID != "call_add" || gotCall.Name != "add" {
		t.Fatalf("call=%+v", gotCall)
	}
	if gotCall.Args["a"] != float64(2) || gotCall.Args["b"] != float64(3) {
		t.Fatalf("args=%v", gotCall.Args)
	}
	if len(seenTools) != 1 {
		t.Fatalf("tools=%d", len(seenTools))
	}
	tool, _ := seenTools[0].(map[string]any)
	params, _ := tool["parameters"].(map[string]any)
	props, _ := params["properties"].(map[string]any)
	if _, ok := props["__aos"]; ok {
		t.Fatalf("codex tool schema should not include __aos: %v", props)
	}
	if len(seenInput) != 3 {
		t.Fatalf("input items=%d", len(seenInput))
	}
}

func TestCodexResponsesModelGenerateImage(t *testing.T) {
	var (
		seenTools      []map[string]any
		seenToolChoice any
	)

	const pngBase64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2Z0ioAAAAASUVORK5CYII="

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var req struct {
			Tools      []map[string]any `json:"tools"`
			ToolChoice any              `json:"tool_choice"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		seenTools = req.Tools
		seenToolChoice = req.ToolChoice

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"ig_1\",\"type\":\"image_generation_call\",\"result\":\"" + pngBase64 + "\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(context.Background(), "gpt-5", &ClientConfig{
		APIKey:     "jwt-token",
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
		Provider:   "codex",
	}, "acct_123")
	if err != nil {
		t.Fatalf("NewCodexResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Generate a landscape cover.", genai.RoleUser),
		},
		Config: &genai.GenerateContentConfig{
			ResponseModalities: []string{"IMAGE"},
			ImageConfig:        &genai.ImageConfig{AspectRatio: "16:9"},
		},
	}
	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	var image *genai.Blob
	for _, part := range responses[0].Content.Parts {
		if part != nil && part.InlineData != nil {
			image = part.InlineData
			break
		}
	}
	if image == nil {
		t.Fatal("expected image output")
	}
	wantBytes, err := base64.StdEncoding.DecodeString(pngBase64)
	if err != nil {
		t.Fatalf("DecodeString: %v", err)
	}
	if image.MIMEType != "image/png" {
		t.Fatalf("mime = %q, want image/png", image.MIMEType)
	}
	if string(image.Data) != string(wantBytes) {
		t.Fatalf("image bytes mismatch")
	}
	if len(seenTools) != 1 {
		t.Fatalf("tools=%d, want 1", len(seenTools))
	}
	if seenTools[0]["type"] != "image_generation" {
		t.Fatalf("tool type = %v", seenTools[0]["type"])
	}
	if seenTools[0]["size"] != "1536x1024" {
		t.Fatalf("tool size = %v", seenTools[0]["size"])
	}
	if seenTools[0]["output_format"] != "png" {
		t.Fatalf("tool output_format = %v", seenTools[0]["output_format"])
	}
	choice, ok := seenToolChoice.(map[string]any)
	if !ok || choice["type"] != "image_generation" {
		t.Fatalf("tool_choice = %#v", seenToolChoice)
	}
}

func collectResponses(seq iter.Seq2[*model.LLMResponse, error]) ([]*model.LLMResponse, error) {
	var out []*model.LLMResponse
	for resp, err := range seq {
		if err != nil {
			return nil, err
		}
		out = append(out, resp)
	}
	return out, nil
}

func extractPartsText(content *genai.Content) string {
	if content == nil {
		return ""
	}
	parts := make([]string, 0, len(content.Parts))
	for _, part := range content.Parts {
		if part == nil || strings.TrimSpace(part.Text) == "" {
			continue
		}
		parts = append(parts, strings.TrimSpace(part.Text))
	}
	return strings.Join(parts, "\n")
}

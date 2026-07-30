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

	providercontract "github.com/cpunion/ailib/adk/model/provider"
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
		seenReasoning codexResponsesReasoning
		seenText      codexResponsesText
		seenInclude   []string
		seenParallel  bool
		seenCacheKey  string
		seenBeta      string
		seenOrigin    string
		seenSession   string
		seenMaxTokens *int
		attempts      []providercontract.ModelAttempt
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
			Model        string                  `json:"model"`
			Store        bool                    `json:"store"`
			Stream       bool                    `json:"stream"`
			Instructions string                  `json:"instructions"`
			Input        []any                   `json:"input"`
			Reasoning    codexResponsesReasoning `json:"reasoning"`
			Text         codexResponsesText      `json:"text"`
			Include      []string                `json:"include"`
			Parallel     bool                    `json:"parallel_tool_calls"`
			CacheKey     string                  `json:"prompt_cache_key"`
			MaxTokens    *int                    `json:"max_output_tokens"`
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
		seenReasoning = req.Reasoning
		seenText = req.Text
		seenInclude = req.Include
		seenParallel = req.Parallel
		seenCacheKey = req.CacheKey
		seenBeta = r.Header.Get("OpenAI-Beta")
		seenOrigin = r.Header.Get("originator")
		seenSession = r.Header.Get("session-id")
		seenMaxTokens = req.MaxTokens

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp-codex-123\",\"model\":\"gpt-5.4-mini\",\"status\":\"in_progress\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\" world\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-codex-123\",\"model\":\"gpt-5.4-mini\",\"status\":\"completed\",\"usage\":{\"input_tokens\":12,\"output_tokens\":5,\"total_tokens\":17}}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey:           "jwt-token",
		BaseURL:          ts.URL,
		HTTPClient:       ts.Client(),
		Provider:         "codex",
		PromptCacheKey:   strings.Repeat("s", 70),
		ReasoningEffort:  "xhigh",
		ReasoningSummary: "auto",
		TextVerbosity:    "low",
		AttemptSink: providercontract.AttemptSinkFunc(func(attempt providercontract.ModelAttempt) {
			attempts = append(attempts, attempt)
		}),
	}, "acct_123")
	if err != nil {
		t.Fatalf("NewCodexResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Say hello.", genai.RoleUser),
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("Evaluation instructions.", "system"),
			MaxOutputTokens:   6000,
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
	if seenPrompt != "Evaluation instructions." {
		t.Fatalf("instructions=%q", seenPrompt)
	}
	if seenReasoning.Effort != "xhigh" || seenReasoning.Summary != "auto" {
		t.Fatalf("reasoning=%+v", seenReasoning)
	}
	if seenText.Verbosity != "low" {
		t.Fatalf("text=%+v", seenText)
	}
	if len(seenInclude) != 1 || seenInclude[0] != "reasoning.encrypted_content" {
		t.Fatalf("include=%v", seenInclude)
	}
	wantSession := strings.Repeat("s", codexSessionIDMaxRunes)
	if !seenParallel || seenCacheKey != wantSession {
		t.Fatalf("parallel=%t cacheKey=%q", seenParallel, seenCacheKey)
	}
	if seenMaxTokens != nil {
		t.Fatalf("codex backend does not accept max_output_tokens=%d", *seenMaxTokens)
	}
	if seenBeta != "responses=experimental" || seenOrigin != "ailib" || seenSession != wantSession {
		t.Fatalf("headers beta=%q origin=%q session=%q", seenBeta, seenOrigin, seenSession)
	}
	if len(seenInput) != 1 {
		t.Fatalf("input items=%d", len(seenInput))
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts=%d", len(attempts))
	}
	attempt := attempts[0]
	if attempt.Provider != "codex" || attempt.Model != "gpt-5.4-mini" || attempt.EndpointKind != providercontract.EndpointKindCodexBackendResponses {
		t.Fatalf("attempt=%+v", attempt)
	}
	if attempt.StatusCode != http.StatusOK || attempt.Usage.TotalTokens != 17 || attempt.EndpointState.CodexAccountID != "acct_123" {
		t.Fatalf("attempt=%+v", attempt)
	}
	if attempt.RequestID != "resp-codex-123" ||
		attempt.ProviderRequestID != "resp-codex-123" {
		t.Fatalf("attempt request identity=%+v", attempt)
	}
}

func TestCodexResponsesModelAttemptSinkRecordsHTTPFailure(t *testing.T) {
	var attempts []providercontract.ModelAttempt
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limited"}}`))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey:     "jwt-token",
		BaseURL:    ts.URL,
		HTTPClient: ts.Client(),
		Provider:   "codex",
		AttemptSink: providercontract.AttemptSinkFunc(func(attempt providercontract.ModelAttempt) {
			attempts = append(attempts, attempt)
		}),
	}, "acct_123")
	if err != nil {
		t.Fatalf("NewCodexResponsesModel: %v", err)
	}

	_, err = collectResponses(llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Say hello.", genai.RoleUser)},
	}, false))
	if err == nil {
		t.Fatalf("expected error")
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts=%d", len(attempts))
	}
	attempt := attempts[0]
	if attempt.StatusCode != http.StatusTooManyRequests || attempt.FailureReason != providercontract.FailoverReasonRateLimit || attempt.ErrorClass != "http_status" {
		t.Fatalf("attempt=%+v", attempt)
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
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.6-sol\",\"status\":\"completed\"}}\n\n"))
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

func TestCodexResponsesModelReasoningItemRoundTrip(t *testing.T) {
	const reasoningItem = `{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"Need call the tool."}],"encrypted_content":"opaque-ciphertext"}`
	var (
		requestCount int
		seenInputs   [][]any
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("ReadAll: %v", err)
		}
		var req struct {
			Input []any `json:"input"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		seenInputs = append(seenInputs, req.Input)

		w.Header().Set("Content-Type", "text/event-stream")
		if requestCount == 1 {
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":" + reasoningItem + "}\n\n"))
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"fc_1\",\"type\":\"function_call\",\"call_id\":\"call_add\",\"name\":\"add\",\"arguments\":\"{\\\"a\\\":2,\\\"b\\\":3}\"}}\n\n"))
		} else {
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"5\"}\n\n"))
		}
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.6-sol\",\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(
		context.Background(),
		"gpt-5.6-sol",
		&ClientConfig{
			APIKey:     "jwt-token",
			BaseURL:    ts.URL,
			HTTPClient: ts.Client(),
			Provider:   "codex",
		},
		"acct_123",
	)
	if err != nil {
		t.Fatal(err)
	}
	first, err := collectResponses(llm.GenerateContent(
		context.Background(),
		&model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("What is 2+3?", genai.RoleUser),
			},
		},
		false,
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 1 || first[0].Content == nil {
		t.Fatalf("first response=%+v", first)
	}
	var (
		thought *genai.Part
		call    *genai.FunctionCall
	)
	for _, part := range first[0].Content.Parts {
		if part.Thought {
			thought = part
		}
		if part.FunctionCall != nil {
			call = part.FunctionCall
		}
	}
	if thought == nil {
		t.Fatal("missing thought")
	}
	replayedItem, signatureOK := decodeResponsesReasoningSignature(
		thought.ThoughtSignature, "codex", "gpt-5.6-sol",
	)
	if !signatureOK ||
		string(replayedItem) != reasoningItem ||
		thought.Text != "Need call the tool." {
		t.Fatalf("thought=%+v", thought)
	}
	if call == nil || call.ID != "call_add" {
		t.Fatalf("call=%+v", call)
	}

	result := genai.NewContentFromFunctionResponse(
		"add",
		map[string]any{"result": 5},
		genai.RoleUser,
	)
	result.Parts[0].FunctionResponse.ID = "call_add"
	_, err = collectResponses(llm.GenerateContent(
		context.Background(),
		&model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("What is 2+3?", genai.RoleUser),
				first[0].Content,
				result,
			},
		},
		false,
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(seenInputs) != 2 {
		t.Fatalf("request inputs=%d", len(seenInputs))
	}
	var replayed map[string]any
	for _, item := range seenInputs[1] {
		object, ok := item.(map[string]any)
		if ok && object["type"] == "reasoning" {
			replayed = object
			break
		}
	}
	if replayed == nil ||
		replayed["id"] != "rs_1" ||
		replayed["encrypted_content"] != "opaque-ciphertext" {
		t.Fatalf("replayed reasoning=%#v input=%#v", replayed, seenInputs[1])
	}
}

func TestCodexResponsesModelPreservesReasoningMessageOrder(t *testing.T) {
	const reasoningItem = `{"id":"rs_order","type":"reasoning","summary":[{"type":"summary_text","text":"Ordered thought."}],"encrypted_content":"ordered-ciphertext"}`
	var (
		requestCount int
		replayInput  []map[string]any
	)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount++
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if requestCount == 2 {
			var req struct {
				Input []map[string]any `json:"input"`
			}
			if err := json.Unmarshal(raw, &req); err != nil {
				t.Fatal(err)
			}
			replayInput = req.Input
		}
		w.Header().Set("Content-Type", "text/event-stream")
		if requestCount == 1 {
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":" + reasoningItem + "}\n\n"))
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_order\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Final answer.\"}]}}\n\n"))
		} else {
			_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"msg_2\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Acknowledged.\"}]}}\n\n"))
		}
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"model\":\"gpt-5.6-sol\",\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewCodexResponsesModel(context.Background(), "gpt-5.6-sol", &ClientConfig{
		APIKey: "jwt-token", BaseURL: ts.URL, HTTPClient: ts.Client(), Provider: "codex",
	}, "acct_123")
	if err != nil {
		t.Fatal(err)
	}
	first, err := collectResponses(llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Answer.", genai.RoleUser)},
	}, false))
	if err != nil {
		t.Fatal(err)
	}
	if len(first) != 1 || first[0].Content == nil || len(first[0].Content.Parts) != 2 {
		t.Fatalf("first=%+v", first)
	}
	if !first[0].Content.Parts[0].Thought ||
		first[0].Content.Parts[1].Text != "Final answer." {
		t.Fatalf("parts=%+v", first[0].Content.Parts)
	}

	_, err = collectResponses(llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{
			genai.NewContentFromText("Answer.", genai.RoleUser),
			first[0].Content,
		},
	}, false))
	if err != nil {
		t.Fatal(err)
	}
	if len(replayInput) < 3 ||
		replayInput[1]["type"] != "reasoning" ||
		replayInput[2]["type"] != "message" {
		t.Fatalf("replay input order=%#v", replayInput)
	}
}

func TestCodexResponsesModelForeignReasoningSignatureFallsBackToText(t *testing.T) {
	otherModelSignature, err := encodeResponsesReasoningSignature(
		"codex",
		"gpt-5.6-terra",
		"gpt-5.6-terra",
		json.RawMessage(
			`{"type":"reasoning","encrypted_content":"model-bound"}`,
		),
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name      string
		signature []byte
	}{
		{
			name:      "foreign provider encoding",
			signature: []byte("anthropic-opaque-signature"),
		},
		{
			name:      "other model envelope",
			signature: otherModelSignature,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			var input []map[string]any
			ts := httptest.NewServer(http.HandlerFunc(func(
				w http.ResponseWriter,
				r *http.Request,
			) {
				raw, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatal(err)
				}
				var req struct {
					Input []map[string]any `json:"input"`
				}
				if err := json.Unmarshal(raw, &req); err != nil {
					t.Fatal(err)
				}
				input = req.Input
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
			}))
			defer ts.Close()

			llm, err := NewCodexResponsesModel(context.Background(), "gpt-5.6-sol", &ClientConfig{
				APIKey: "jwt-token", BaseURL: ts.URL,
				HTTPClient: ts.Client(), Provider: "codex",
			}, "acct_123")
			if err != nil {
				t.Fatal(err)
			}
			foreign := &genai.Part{
				Text: "portable reasoning summary", Thought: true,
				ThoughtSignature: test.signature,
			}
			_, err = collectResponses(llm.GenerateContent(context.Background(), &model.LLMRequest{
				Contents: []*genai.Content{{Role: "model", Parts: []*genai.Part{foreign}}},
			}, false))
			if err != nil {
				t.Fatalf("foreign signature must degrade, got %v", err)
			}
			if len(input) < 1 || input[0]["type"] != "message" {
				t.Fatalf("input=%#v", input)
			}
			content, _ := input[0]["content"].([]any)
			if len(content) != 1 {
				t.Fatalf("message content=%#v", input[0]["content"])
			}
			text, _ := content[0].(map[string]any)
			if text["text"] != "portable reasoning summary" {
				t.Fatalf("fallback=%#v", input)
			}
		})
	}
}

func TestResponsesReasoningSignatureRejectsUnstableRoute(t *testing.T) {
	item := json.RawMessage(
		`{"type":"reasoning","encrypted_content":"model-bound"}`,
	)
	for _, test := range []struct {
		name          string
		provider      string
		requestModel  string
		responseModel string
		wantReplay    bool
	}{
		{
			name:     "openrouter auto",
			provider: "openrouter", requestModel: "auto",
			responseModel: "anthropic/claude-sonnet-4", wantReplay: false,
		},
		{
			name:     "openrouter auto without effective model",
			provider: "openrouter", requestModel: "auto",
			responseModel: "", wantReplay: false,
		},
		{
			name:     "openrouter alias",
			provider: "openrouter", requestModel: "openai/gpt-5",
			responseModel: "openai/gpt-5-2026-07-30", wantReplay: false,
		},
		{
			name:     "openai unrelated response",
			provider: "openai", requestModel: "gpt-5",
			responseModel: "gpt-4.1-2025-04-14", wantReplay: false,
		},
		{
			name:     "openai versioned response",
			provider: "openai", requestModel: "gpt-5",
			responseModel: "gpt-5-2026-07-30", wantReplay: true,
		},
		{
			name:     "codex versioned response",
			provider: "codex", requestModel: "gpt-5.6-sol",
			responseModel: "gpt-5.6-sol-2026-07-30", wantReplay: true,
		},
		{
			name:     "exact route",
			provider: "openrouter", requestModel: "openai/gpt-5",
			responseModel: "openai/gpt-5", wantReplay: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			signature, err := encodeResponsesReasoningSignature(
				test.provider,
				test.requestModel,
				test.responseModel,
				item,
			)
			if err != nil {
				t.Fatal(err)
			}
			_, replay := decodeResponsesReasoningSignature(
				signature,
				test.provider,
				test.requestModel,
			)
			if replay != test.wantReplay {
				t.Fatalf(
					"replay=%v want=%v signature=%s",
					replay,
					test.wantReplay,
					signature,
				)
			}
		})
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

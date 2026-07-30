package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	providercontract "github.com/cpunion/ailib/adk/model/provider"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestResponsesModelGenerateText(t *testing.T) {
	var (
		seenPath    string
		seenAuth    string
		seenAccount string
		seenModel   string
		seenStream  bool
		seenPrompt  string
		seenCache   string
		seenInclude []string
		seenInput   []any
		attempts    []providercontract.ModelAttempt
	)

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Model          string   `json:"model"`
			Stream         bool     `json:"stream"`
			Instructions   string   `json:"instructions"`
			PromptCacheKey string   `json:"prompt_cache_key"`
			Include        []string `json:"include"`
			Input          []any    `json:"input"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		seenPath = r.URL.Path
		seenAuth = r.Header.Get("Authorization")
		seenAccount = r.Header.Get("chatgpt-account-id")
		seenModel = req.Model
		seenStream = req.Stream
		seenPrompt = req.Instructions
		seenCache = req.PromptCacheKey
		seenInclude = req.Include
		seenInput = req.Input

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\" world\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp-openai-123\",\"model\":\"gpt-5.4-mini\",\"status\":\"completed\",\"usage\":{\"input_tokens\":12,\"output_tokens\":5,\"total_tokens\":17,\"input_tokens_details\":{\"cached_tokens\":4}}}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey:         "sk-test",
		BaseURL:        ts.URL,
		HTTPClient:     ts.Client(),
		Provider:       "openai",
		PromptCacheKey: "sess-cache-key",
		AttemptSink: providercontract.AttemptSinkFunc(func(a providercontract.ModelAttempt) {
			attempts = append(attempts, a)
		}),
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Say hello.", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: genai.NewContentFromText("You are helpful.", "system"),
		},
	}
	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	if got := extractPartsText(responses[0].Content); got != "Hello world" {
		t.Fatalf("text=%q", got)
	}
	if seenPath != "/responses" {
		t.Fatalf("path=%q", seenPath)
	}
	if seenAuth != "Bearer sk-test" {
		t.Fatalf("auth=%q", seenAuth)
	}
	if seenAccount != "" {
		t.Fatalf("generic responses must not send chatgpt-account-id, got %q", seenAccount)
	}
	if seenModel != "gpt-5.4-mini" || !seenStream {
		t.Fatalf("model=%q stream=%v", seenModel, seenStream)
	}
	if !strings.Contains(seenPrompt, "You are helpful.") {
		t.Fatalf("instructions=%q", seenPrompt)
	}
	if seenCache != "sess-cache-key" {
		t.Fatalf("prompt_cache_key=%q want sess-cache-key", seenCache)
	}
	if len(seenInclude) != 1 ||
		seenInclude[0] != "reasoning.encrypted_content" {
		t.Fatalf("include=%v", seenInclude)
	}
	if len(seenInput) != 1 {
		t.Fatalf("input items=%d", len(seenInput))
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts=%d", len(attempts))
	}
	a := attempts[0]
	if a.Provider != "openai" || a.EndpointKind != providercontract.EndpointKindResponses {
		t.Fatalf("attempt provider/kind=%+v", a)
	}
	if a.RequestID != "resp-openai-123" ||
		a.ProviderRequestID != "resp-openai-123" {
		t.Fatalf("attempt request identity=%+v", a)
	}
	if a.StatusCode != http.StatusOK || a.Usage.TotalTokens != 17 || a.Usage.Cache.ReadTokens != 4 || !a.Usage.Cache.Hit {
		t.Fatalf("attempt=%+v", a)
	}
}

func TestResponsesModelRequestScopedCacheKeyOverridesConfig(t *testing.T) {
	var seenCache string
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			PromptCacheKey string `json:"prompt_cache_key"`
		}
		_ = json.Unmarshal(raw, &req)
		seenCache = req.PromptCacheKey
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(), PromptCacheKey: "client-default",
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}
	ctx := providercontract.WithPromptCacheKey(context.Background(), "per-request-key")
	if _, err := collectResponses(llm.GenerateContent(ctx, &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false)); err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if seenCache != "per-request-key" {
		t.Fatalf("prompt_cache_key=%q want per-request-key", seenCache)
	}
}

func TestResponsesModelEncryptedReasoningIncludeByProvider(t *testing.T) {
	for _, tc := range []struct {
		provider string
		want     bool
	}{
		{provider: "openai", want: true},
		{provider: "openrouter", want: true},
		{provider: "groq", want: false},
	} {
		t.Run(tc.provider, func(t *testing.T) {
			m := &responsesModel{
				modelName: "test-model",
				config:    &ClientConfig{Provider: tc.provider},
			}
			request, err := m.convertRequest(&model.LLMRequest{})
			if err != nil {
				t.Fatalf("convertRequest: %v", err)
			}
			got := len(request.Include) == 1 &&
				request.Include[0] == "reasoning.encrypted_content"
			if got != tc.want {
				t.Fatalf("include=%v wantEncryptedReasoning=%v", request.Include, tc.want)
			}
		})
	}
}

func TestResponsesModelFunctionCallRoundTrip(t *testing.T) {
	var seenInput []any
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Input []any `json:"input"`
			Tools []any `json:"tools"`
		}
		_ = json.Unmarshal(raw, &req)
		seenInput = req.Input

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_abc\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"SF\\\"}\"}}\n\n"))
		_, _ = w.Write([]byte("data: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n"))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(),
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}

	req := &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("Weather in SF?", genai.RoleUser)},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{{
				FunctionDeclarations: []*genai.FunctionDeclaration{{
					Name:        "get_weather",
					Description: "Get weather",
					Parameters:  &genai.Schema{Type: genai.TypeObject},
				}},
			}},
		},
	}
	responses, err := collectResponses(llm.GenerateContent(context.Background(), req, false))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if len(responses) != 1 {
		t.Fatalf("responses=%d", len(responses))
	}
	var fc *genai.FunctionCall
	for _, p := range responses[0].Content.Parts {
		if p.FunctionCall != nil {
			fc = p.FunctionCall
		}
	}
	if fc == nil {
		t.Fatal("expected a function call part")
	}
	if fc.Name != "get_weather" || fc.ID != "call_abc" {
		t.Fatalf("function call=%+v", fc)
	}
	if city, _ := fc.Args["city"].(string); city != "SF" {
		t.Fatalf("args=%+v", fc.Args)
	}
	if len(seenInput) != 1 {
		t.Fatalf("input items=%d", len(seenInput))
	}
}

func TestResponsesModelAttemptSinkRecordsHTTPFailure(t *testing.T) {
	var attempts []providercontract.ModelAttempt
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"slow down"}}`))
	}))
	defer ts.Close()

	llm, err := NewResponsesModel(context.Background(), "gpt-5.4-mini", &ClientConfig{
		APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(),
		AttemptSink: providercontract.AttemptSinkFunc(func(a providercontract.ModelAttempt) {
			attempts = append(attempts, a)
		}),
	})
	if err != nil {
		t.Fatalf("NewResponsesModel: %v", err)
	}
	_, err = collectResponses(llm.GenerateContent(context.Background(), &model.LLMRequest{
		Contents: []*genai.Content{genai.NewContentFromText("hi", genai.RoleUser)},
	}, false))
	if err == nil {
		t.Fatal("expected error on 429")
	}
	if len(attempts) != 1 {
		t.Fatalf("attempts=%d", len(attempts))
	}
	a := attempts[0]
	if a.StatusCode != http.StatusTooManyRequests || a.FailureReason != providercontract.FailoverReasonRateLimit {
		t.Fatalf("attempt=%+v", a)
	}
	if a.EndpointKind != providercontract.EndpointKindResponses {
		t.Fatalf("endpointKind=%v", a.EndpointKind)
	}
}

func TestResponsesModelRejectsMissingTerminalEvent(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		_ *http.Request,
	) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n",
		))
	}))
	defer ts.Close()
	llm, err := NewResponsesModel(
		context.Background(),
		"gpt-5.6-terra",
		&ClientConfig{
			APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := collectResponses(llm.GenerateContent(
		context.Background(),
		&model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("hi", genai.RoleUser),
			},
		},
		false,
	)); err == nil || !strings.Contains(err.Error(), "terminal event") {
		t.Fatalf("error=%v", err)
	}
}

func TestResponsesModelIncompleteIsMaxTokens(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		_ *http.Request,
	) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(
			"data: {\"type\":\"response.incomplete\",\"response\":{\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
		))
	}))
	defer ts.Close()
	llm, err := NewResponsesModel(
		context.Background(),
		"gpt-5.6-terra",
		&ClientConfig{
			APIKey: "sk-test", BaseURL: ts.URL, HTTPClient: ts.Client(),
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	responses, err := collectResponses(llm.GenerateContent(
		context.Background(),
		&model.LLMRequest{
			Contents: []*genai.Content{
				genai.NewContentFromText("hi", genai.RoleUser),
			},
		},
		false,
	))
	if err != nil {
		t.Fatal(err)
	}
	if len(responses) != 1 ||
		responses[0].FinishReason != genai.FinishReasonMaxTokens {
		t.Fatalf("responses=%+v", responses)
	}
}

func TestResponsesModelBackfillsOnlyMissingTerminalEncryptedReasoning(
	t *testing.T,
) {
	for _, test := range []struct {
		name              string
		doneEncrypted     string
		terminalEncrypted string
		wantEncrypted     string
	}{
		{
			name:              "terminal backfill",
			terminalEncrypted: "ENC-FINAL",
			wantEncrypted:     "ENC-FINAL",
		},
		{
			name:              "completed item remains authoritative",
			doneEncrypted:     "ENC-DONE",
			terminalEncrypted: "ENC-TERMINAL",
			wantEncrypted:     "ENC-DONE",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			doneEncrypted := ""
			if test.doneEncrypted != "" {
				doneEncrypted =
					`,"encrypted_content":"` + test.doneEncrypted + `"`
			}
			ts := httptest.NewServer(http.HandlerFunc(func(
				w http.ResponseWriter,
				_ *http.Request,
			) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(
					w,
					`data: {"type":"response.output_item.done","item":{"id":"rs_1","type":"reasoning","summary":[{"type":"summary_text","text":"inspect"}],"provider_extension":{"v":1}`+
						doneEncrypted+"}}\n\n",
				)
				_, _ = io.WriteString(
					w,
					`data: {"type":"response.completed","response":{"model":"gpt-5","status":"completed","output":[{"id":"rs_1","type":"reasoning","encrypted_content":"`+
						test.terminalEncrypted+
						`","provider_extension":{"v":2}}]}}`+"\n\n",
				)
			}))
			defer ts.Close()

			llm, err := NewResponsesModel(
				context.Background(),
				"gpt-5",
				&ClientConfig{
					APIKey:     "sk-test",
					BaseURL:    ts.URL,
					HTTPClient: ts.Client(),
					Provider:   "openai",
				},
			)
			if err != nil {
				t.Fatal(err)
			}
			responses, err := collectResponses(llm.GenerateContent(
				context.Background(),
				&model.LLMRequest{
					Contents: []*genai.Content{
						genai.NewContentFromText(
							"inspect",
							genai.RoleUser,
						),
					},
				},
				false,
			))
			if err != nil {
				t.Fatal(err)
			}
			if len(responses) != 1 ||
				responses[0].Content == nil ||
				len(responses[0].Content.Parts) != 1 {
				t.Fatalf("responses=%+v", responses)
			}
			part := responses[0].Content.Parts[0]
			raw, ok := decodeResponsesReasoningSignature(
				part.ThoughtSignature,
				"openai",
				"gpt-5",
			)
			if !ok {
				t.Fatalf("signature=%s", part.ThoughtSignature)
			}
			var item map[string]any
			if err := json.Unmarshal(raw, &item); err != nil {
				t.Fatal(err)
			}
			extension, _ := item["provider_extension"].(map[string]any)
			if item["encrypted_content"] != test.wantEncrypted ||
				extension["v"] != float64(1) {
				t.Fatalf("item=%#v", item)
			}
		})
	}
}

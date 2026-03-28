package model

import (
	"context"
	"errors"
	"testing"

	"iter"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

func TestWrapLLM_ObservesResponsesAndDone(t *testing.T) {
	var (
		started bool
		done    bool
		indexes []int
	)
	llm := WrapLLM(NewMockLLM(
		&adkmodel.LLMResponse{Content: genai.NewContentFromText("hello", genai.RoleModel), Partial: true},
		&adkmodel.LLMResponse{Content: genai.NewContentFromText("world", genai.RoleModel), Partial: false, TurnComplete: true},
	), GenerateHooks{
		OnStart: func(_ context.Context, req *adkmodel.LLMRequest, stream bool) any {
			started = req != nil && stream
			return "state"
		},
		OnResponse: func(state any, index int, resp *adkmodel.LLMResponse) {
			if state != "state" {
				t.Fatalf("state=%v", state)
			}
			indexes = append(indexes, index)
			if resp.CustomMetadata == nil {
				resp.CustomMetadata = map[string]any{}
			}
			resp.CustomMetadata["wrapped"] = true
		},
		OnDone: func(state any) {
			if state != "state" {
				t.Fatalf("state=%v", state)
			}
			done = true
		},
	})

	resp, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{Model: "test"}, true))
	if err != nil {
		t.Fatalf("GenerateContent: %v", err)
	}
	if !started || !done {
		t.Fatalf("started=%v done=%v", started, done)
	}
	if len(indexes) != 2 || indexes[0] != 0 || indexes[1] != 1 {
		t.Fatalf("indexes=%v", indexes)
	}
	if len(resp) != 2 || resp[0].CustomMetadata["wrapped"] != true || resp[1].CustomMetadata["wrapped"] != true {
		t.Fatalf("responses not wrapped: %+v", resp)
	}
}

func TestWrapLLM_ObservesErrors(t *testing.T) {
	wantErr := errors.New("boom")
	var seenErr error
	llm := WrapLLM(errorLLM{err: wantErr}, GenerateHooks{
		OnError: func(_ any, err error) {
			seenErr = err
		},
	})

	if _, err := readAll(llm.GenerateContent(context.Background(), &adkmodel.LLMRequest{Model: "test"}, true)); !errors.Is(err, wantErr) {
		t.Fatalf("err=%v", err)
	}
	if !errors.Is(seenErr, wantErr) {
		t.Fatalf("seenErr=%v", seenErr)
	}
}

type errorLLM struct {
	err error
}

func (e errorLLM) Name() string { return "error-llm" }

func (e errorLLM) GenerateContent(_ context.Context, _ *adkmodel.LLMRequest, _ bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		yield(nil, e.err)
	}
}

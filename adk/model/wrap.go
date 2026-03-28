package model

import (
	"context"
	"iter"

	adkmodel "google.golang.org/adk/model"
)

// GenerateHooks allows callers to observe and optionally mutate GenerateContent
// request/response flow without reimplementing an LLM wrapper each time.
type GenerateHooks struct {
	OnStart    func(ctx context.Context, req *adkmodel.LLMRequest, stream bool) any
	OnResponse func(state any, index int, resp *adkmodel.LLMResponse)
	OnError    func(state any, err error)
	OnDone     func(state any)
}

type wrappedLLM struct {
	inner adkmodel.LLM
	hooks GenerateHooks
}

// WrapLLM applies lightweight GenerateContent hooks around an LLM.
func WrapLLM(inner adkmodel.LLM, hooks GenerateHooks) adkmodel.LLM {
	if inner == nil {
		return nil
	}
	return wrappedLLM{inner: inner, hooks: hooks}
}

func (w wrappedLLM) Name() string {
	return w.inner.Name()
}

func (w wrappedLLM) GenerateContent(ctx context.Context, req *adkmodel.LLMRequest, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	state := any(nil)
	if w.hooks.OnStart != nil {
		state = w.hooks.OnStart(ctx, req, stream)
	}
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		index := 0
		for resp, err := range w.inner.GenerateContent(ctx, req, stream) {
			if err != nil {
				if w.hooks.OnError != nil {
					w.hooks.OnError(state, err)
				}
				yield(nil, err)
				return
			}
			if resp != nil && w.hooks.OnResponse != nil {
				w.hooks.OnResponse(state, index, resp)
			}
			index++
			if !yield(resp, nil) {
				if w.hooks.OnDone != nil {
					w.hooks.OnDone(state)
				}
				return
			}
		}
		if w.hooks.OnDone != nil {
			w.hooks.OnDone(state)
		}
	}
}

var _ adkmodel.LLM = (*wrappedLLM)(nil)

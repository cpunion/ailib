package model

import (
	"context"
	"iter"

	adkmodel "google.golang.org/adk/model"
)

// MockLLM is a simplified mock LLM for testing.
// It yields all Responses in order when GenerateContent is called with stream=true.
// For multi-turn conversations, use MockConversation instead.
type MockLLM struct {
	// Responses to yield for this single turn (supports partial and thought content)
	Responses []*adkmodel.LLMResponse
}

// NewMockLLM creates a mock LLM with the given responses for one turn.
func NewMockLLM(responses ...*adkmodel.LLMResponse) *MockLLM {
	return &MockLLM{Responses: responses}
}

// Name implements model.LLM.
func (m *MockLLM) Name() string {
	return "mock-llm"
}

// GenerateContent implements model.LLM.
// It yields all responses in order when streaming, or just the last one when not streaming.
func (m *MockLLM) GenerateContent(_ context.Context, _ *adkmodel.LLMRequest, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		if len(m.Responses) == 0 {
			return
		}
		if stream {
			for _, resp := range m.Responses {
				if !yield(resp, nil) {
					return
				}
			}
		} else {
			// Non-streaming: yield only the last response (final)
			yield(m.Responses[len(m.Responses)-1], nil)
		}
	}
}

var _ adkmodel.LLM = (*MockLLM)(nil)

// --- Multi-turn conversation testing utilities ---

// Turn represents a single turn in a conversation.
type Turn struct {
	Request   *adkmodel.LLMRequest
	Responses []*adkmodel.LLMResponse
}

// MockConversation provides multi-turn conversation testing.
// Each GenerateContent call consumes the next turn.
type MockConversation struct {
	Turns   []*Turn
	current int
}

// NewMockConversation creates a mock conversation with multiple turns.
func NewMockConversation(turns ...*Turn) *MockConversation {
	return &MockConversation{Turns: turns}
}

// Name implements model.LLM.
func (m *MockConversation) Name() string {
	return "mock-conversation"
}

// GenerateContent implements model.LLM.
// It yields responses for the current turn, then advances to the next turn.
func (m *MockConversation) GenerateContent(_ context.Context, _ *adkmodel.LLMRequest, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		if m.current >= len(m.Turns) {
			return
		}
		turn := m.Turns[m.current]
		m.current++

		if len(turn.Responses) == 0 {
			return
		}
		if stream {
			for _, resp := range turn.Responses {
				if !yield(resp, nil) {
					return
				}
			}
		} else {
			yield(turn.Responses[len(turn.Responses)-1], nil)
		}
	}
}

// Reset resets the conversation to the first turn.
func (m *MockConversation) Reset() {
	m.current = 0
}

var _ adkmodel.LLM = (*MockConversation)(nil)

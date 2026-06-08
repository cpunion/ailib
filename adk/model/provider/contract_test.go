package provider

import (
	"context"
	"errors"
	"testing"
)

func TestBaseURLClass(t *testing.T) {
	tests := []struct {
		raw  string
		want string
	}{
		{raw: "https://api.openai.com/v1", want: "openai"},
		{raw: "https://openrouter.ai/api/v1", want: "openrouter"},
		{raw: "https://api.deepseek.com/v1", want: "deepseek"},
		{raw: "https://api.groq.com/openai/v1", want: "groq"},
		{raw: "https://chatgpt.com/backend-api/codex", want: "chatgpt"},
		{raw: "https://example.invalid/v1", want: "custom"},
		{raw: "://bad", want: "invalid"},
		{raw: "", want: ""},
	}
	for _, tt := range tests {
		if got := BaseURLClass(tt.raw); got != tt.want {
			t.Fatalf("BaseURLClass(%q) = %q, want %q", tt.raw, got, tt.want)
		}
	}
}

func TestClassifyHTTPStatus(t *testing.T) {
	tests := []struct {
		status int
		want   FailoverReason
	}{
		{status: 200, want: FailoverReasonNone},
		{status: 401, want: FailoverReasonAuth},
		{status: 403, want: FailoverReasonAuth},
		{status: 408, want: FailoverReasonTimeout},
		{status: 413, want: FailoverReasonContext},
		{status: 429, want: FailoverReasonRateLimit},
		{status: 500, want: FailoverReasonProviderBug},
		{status: 400, want: FailoverReasonHTTPStatus},
	}
	for _, tt := range tests {
		if got := ClassifyHTTPStatus(tt.status); got != tt.want {
			t.Fatalf("ClassifyHTTPStatus(%d) = %q, want %q", tt.status, got, tt.want)
		}
	}
}

func TestClassifyError(t *testing.T) {
	tests := []struct {
		err  error
		want FailoverReason
	}{
		{err: nil, want: FailoverReasonNone},
		{err: context.Canceled, want: FailoverReasonCanceled},
		{err: context.DeadlineExceeded, want: FailoverReasonTimeout},
		{err: errors.New("failed to decode response"), want: FailoverReasonDecode},
		{err: errors.New("tool schema rejected"), want: FailoverReasonToolSchema},
		{err: errors.New("something else"), want: FailoverReasonUnknown},
	}
	for _, tt := range tests {
		if got := ClassifyError(tt.err); got != tt.want {
			t.Fatalf("ClassifyError(%v) = %q, want %q", tt.err, got, tt.want)
		}
	}
}

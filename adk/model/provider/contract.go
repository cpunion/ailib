// Package provider defines provider-neutral model call contracts.
package provider

import (
	"context"
	"errors"
	"net"
	"net/url"
	"strings"
	"time"
)

// EndpointKind identifies a provider API shape.
type EndpointKind string

const (
	EndpointKindChatCompletions       EndpointKind = "chat_completions"
	EndpointKindResponses             EndpointKind = "responses"
	EndpointKindGeminiGenerateContent EndpointKind = "gemini_generate_content"
	EndpointKindGeminiInteractions    EndpointKind = "gemini_interactions"
	EndpointKindGeminiLive            EndpointKind = "gemini_live"
	EndpointKindCodexBackendResponses EndpointKind = "codex_backend_responses"
)

// FailoverReason is a normalized provider failure class.
type FailoverReason string

const (
	FailoverReasonNone        FailoverReason = ""
	FailoverReasonAuth        FailoverReason = "auth"
	FailoverReasonRateLimit   FailoverReason = "rate_limit"
	FailoverReasonContext     FailoverReason = "context"
	FailoverReasonTimeout     FailoverReason = "timeout"
	FailoverReasonEmpty       FailoverReason = "empty"
	FailoverReasonToolSchema  FailoverReason = "tool_schema"
	FailoverReasonNoBackend   FailoverReason = "no_backend"
	FailoverReasonProviderBug FailoverReason = "provider_bug"
	FailoverReasonCanceled    FailoverReason = "canceled"
	FailoverReasonHTTPStatus  FailoverReason = "http_status"
	FailoverReasonDecode      FailoverReason = "decode"
	FailoverReasonNetwork     FailoverReason = "network"
	FailoverReasonUnknown     FailoverReason = "unknown"
)

// CacheUsage normalizes provider prompt/context cache accounting.
type CacheUsage struct {
	ReadTokens   int64         `json:"readTokens,omitempty"`
	WriteTokens  int64         `json:"writeTokens,omitempty"`
	CreateTokens int64         `json:"createTokens,omitempty"`
	TTL          time.Duration `json:"ttl,omitempty"`
	CacheKey     string        `json:"cacheKey,omitempty"`
	ProviderKey  string        `json:"providerKey,omitempty"`
	Hit          bool          `json:"hit,omitempty"`
	Eligibility  string        `json:"eligibility,omitempty"`
	MissReason   string        `json:"missReason,omitempty"`
}

// Usage normalizes token, cache, and provider cost accounting.
type Usage struct {
	InputTokens     int64      `json:"inputTokens,omitempty"`
	OutputTokens    int64      `json:"outputTokens,omitempty"`
	TotalTokens     int64      `json:"totalTokens,omitempty"`
	ReasoningTokens int64      `json:"reasoningTokens,omitempty"`
	AudioTokens     int64      `json:"audioTokens,omitempty"`
	ImageTokens     int64      `json:"imageTokens,omitempty"`
	Cache           CacheUsage `json:"cache,omitempty"`
	CostCredits     float64    `json:"costCredits,omitempty"`
}

// EndpointState stores provider-side continuation or trace identifiers.
type EndpointState struct {
	ResponseID             string `json:"responseId,omitempty"`
	InteractionID          string `json:"interactionId,omitempty"`
	PreviousInteractionID  string `json:"previousInteractionId,omitempty"`
	CodexAccountID         string `json:"codexAccountId,omitempty"`
	OpenRouterGenerationID string `json:"openRouterGenerationId,omitempty"`
}

// RawTrace contains redacted diagnostic samples. Callers should leave it empty
// unless a diagnostic mode explicitly permits storing raw provider payloads.
type RawTrace struct {
	RequestSample  string `json:"requestSample,omitempty"`
	ResponseSample string `json:"responseSample,omitempty"`
	ErrorSample    string `json:"errorSample,omitempty"`
}

// ModelAttempt describes one provider call attempt.
type ModelAttempt struct {
	Provider           string         `json:"provider,omitempty"`
	Model              string         `json:"model,omitempty"`
	EndpointKind       EndpointKind   `json:"endpointKind,omitempty"`
	BaseURLClass       string         `json:"baseUrlClass,omitempty"`
	RequestID          string         `json:"requestId,omitempty"`
	ProviderRequestID  string         `json:"providerRequestId,omitempty"`
	StatusCode         int            `json:"statusCode,omitempty"`
	FinishReason       string         `json:"finishReason,omitempty"`
	NativeFinishReason string         `json:"nativeFinishReason,omitempty"`
	FailureReason      FailoverReason `json:"failureReason,omitempty"`
	ErrorClass         string         `json:"errorClass,omitempty"`
	Usage              Usage          `json:"usage,omitempty"`
	Cache              CacheUsage     `json:"cache,omitempty"`
	EndpointState      EndpointState  `json:"endpointState,omitempty"`
	RawTrace           RawTrace       `json:"rawTrace,omitempty"`
	LatencyMS          int64          `json:"latencyMs,omitempty"`
	TimeToFirstTokenMS int64          `json:"timeToFirstTokenMs,omitempty"`
	StreamChunkCount   int            `json:"streamChunkCount,omitempty"`
}

// AttemptSink observes provider attempts.
type AttemptSink interface {
	ObserveModelAttempt(ModelAttempt)
}

// AttemptSinkFunc adapts a function to AttemptSink.
type AttemptSinkFunc func(ModelAttempt)

// ObserveModelAttempt implements AttemptSink.
func (f AttemptSinkFunc) ObserveModelAttempt(attempt ModelAttempt) {
	if f != nil {
		f(attempt)
	}
}

// BaseURLClass returns a redacted base URL class suitable for metrics.
func BaseURLClass(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	u, err := url.Parse(raw)
	if err != nil || strings.TrimSpace(u.Host) == "" {
		return "invalid"
	}
	host := strings.ToLower(strings.TrimSpace(u.Hostname()))
	switch {
	case host == "api.openai.com":
		return "openai"
	case host == "openrouter.ai":
		return "openrouter"
	case host == "api.deepseek.com":
		return "deepseek"
	case host == "api.minimax.io":
		return "minimax"
	case host == "api.groq.com":
		return "groq"
	case host == "chatgpt.com":
		return "chatgpt"
	default:
		return "custom"
	}
}

// ClassifyHTTPStatus maps HTTP status codes to normalized failure reasons.
func ClassifyHTTPStatus(status int) FailoverReason {
	switch {
	case status == 0:
		return FailoverReasonUnknown
	case status == 401 || status == 403:
		return FailoverReasonAuth
	case status == 408 || status == 504:
		return FailoverReasonTimeout
	case status == 413:
		return FailoverReasonContext
	case status == 429:
		return FailoverReasonRateLimit
	case status >= 500:
		return FailoverReasonProviderBug
	case status >= 400:
		return FailoverReasonHTTPStatus
	default:
		return FailoverReasonNone
	}
}

// ClassifyError maps common Go errors to normalized failure reasons.
func ClassifyError(err error) FailoverReason {
	if err == nil {
		return FailoverReasonNone
	}
	if errors.Is(err, context.Canceled) {
		return FailoverReasonCanceled
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return FailoverReasonTimeout
	}
	var netErr net.Error
	if errors.As(err, &netErr) {
		if netErr.Timeout() {
			return FailoverReasonTimeout
		}
		return FailoverReasonNetwork
	}
	msg := strings.ToLower(err.Error())
	switch {
	case strings.Contains(msg, "context canceled"):
		return FailoverReasonCanceled
	case strings.Contains(msg, "deadline exceeded"), strings.Contains(msg, "timeout"):
		return FailoverReasonTimeout
	case strings.Contains(msg, "json"), strings.Contains(msg, "decode"), strings.Contains(msg, "unmarshal"):
		return FailoverReasonDecode
	case strings.Contains(msg, "schema"), strings.Contains(msg, "tool"):
		return FailoverReasonToolSchema
	default:
		return FailoverReasonUnknown
	}
}

package provider

import (
	"context"
	"strings"
)

// Prompt cache key request-scoped override: model clients are often shared
// across sessions, while prompt cache bucketing wants a per-session (or
// per-agent) key. Callers attach the key to the request context; providers
// that support native prompt cache bucketing read it at request build time,
// falling back to their client-level configuration.

type promptCacheKeyContextKey struct{}

// WithPromptCacheKey returns a context carrying a prompt cache bucketing key
// for this request.
func WithPromptCacheKey(ctx context.Context, key string) context.Context {
	key = strings.TrimSpace(key)
	if key == "" {
		return ctx
	}
	return context.WithValue(ctx, promptCacheKeyContextKey{}, key)
}

// PromptCacheKeyFromContext returns the request-scoped prompt cache key, if
// any.
func PromptCacheKeyFromContext(ctx context.Context) string {
	if ctx == nil {
		return ""
	}
	if v, ok := ctx.Value(promptCacheKeyContextKey{}).(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

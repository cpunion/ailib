package provider

import (
	"context"
	"testing"
)

func TestPromptCacheKeyContextRoundTrip(t *testing.T) {
	ctx := context.Background()
	if got := PromptCacheKeyFromContext(ctx); got != "" {
		t.Fatalf("empty context should carry no key, got %q", got)
	}
	ctx = WithPromptCacheKey(ctx, " session-1 ")
	if got := PromptCacheKeyFromContext(ctx); got != "session-1" {
		t.Fatalf("key round trip = %q, want session-1", got)
	}
	if same := WithPromptCacheKey(context.Background(), "  "); PromptCacheKeyFromContext(same) != "" {
		t.Fatalf("blank keys must not be stored")
	}
}

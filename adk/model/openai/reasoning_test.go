package openai

import "testing"

func TestResolveReasoningEffortCodex(t *testing.T) {
	if got := resolveReasoningEffort("codex", "Runtime: repo=/workspace | thinking=xhigh"); got != "xhigh" {
		t.Fatalf("effort=%q", got)
	}
	if got := resolveReasoningEffort("codex", "Runtime: repo=/workspace | thinking=extra-high"); got != "xhigh" {
		t.Fatalf("effort=%q", got)
	}
	if got := resolveReasoningEffort("openai", "Runtime: repo=/workspace | thinking=xhigh"); got != "" {
		t.Fatalf("effort=%q", got)
	}
}

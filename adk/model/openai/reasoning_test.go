package openai

import (
	"strings"
	"testing"
	"unicode/utf8"
)

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

func TestNormalizeCodexRequestOptionsRejectsUnknownValues(t *testing.T) {
	if _, err := normalizeCodexReasoningEffort("ultra"); err == nil {
		t.Fatal("unknown reasoning effort must fail closed")
	}
	if _, err := normalizeCodexReasoningSummary("verbose"); err == nil {
		t.Fatal("unknown reasoning summary must fail closed")
	}
	if _, err := normalizeCodexTextVerbosity("maximum"); err == nil {
		t.Fatal("unknown text verbosity must fail closed")
	}
}

func TestClampCodexSessionIDUsesRuneLimit(t *testing.T) {
	got := clampCodexSessionID(strings.Repeat("中", 70))
	if utf8.RuneCountInString(got) != codexSessionIDMaxRunes {
		t.Fatalf("runes=%d want=%d", utf8.RuneCountInString(got), codexSessionIDMaxRunes)
	}
	if !utf8.ValidString(got) {
		t.Fatal("clamped session id is invalid UTF-8")
	}
}

package model

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"strings"
	"time"

	adkmodel "google.golang.org/adk/model"
	"google.golang.org/genai"
)

const (
	defaultMockEchoRepeat = 1
	defaultMockEchoDelay  = 100 * time.Millisecond
	defaultMockEchoPrefix = "echo: "
)

// MockEchoSpec configures mock echo behavior.
type MockEchoSpec struct {
	Repeat int    `json:"repeat,omitempty"`
	Delay  string `json:"delay,omitempty"`
	Prefix string `json:"prefix,omitempty"`
}

// MockEchoConfig is the normalized runtime config.
type MockEchoConfig struct {
	Repeat int
	Delay  time.Duration
	Prefix string
}

func defaultMockEchoConfig() MockEchoConfig {
	return MockEchoConfig{
		Repeat: defaultMockEchoRepeat,
		Delay:  defaultMockEchoDelay,
		Prefix: defaultMockEchoPrefix,
	}
}

func normalizeMockEchoConfig(cfg MockEchoConfig) (MockEchoConfig, error) {
	if cfg.Repeat == 0 {
		cfg.Repeat = defaultMockEchoRepeat
	}
	if strings.TrimSpace(cfg.Prefix) == "" {
		cfg.Prefix = defaultMockEchoPrefix
	}
	if cfg.Repeat <= 0 {
		return MockEchoConfig{}, fmt.Errorf("mock-echo repeat must be > 0")
	}
	if cfg.Delay < 0 {
		return MockEchoConfig{}, fmt.Errorf("mock-echo delay must be >= 0")
	}
	return cfg, nil
}

// ParseMockEchoConfig parses the JSON object payload after mock-echo or mock:echo.
func ParseMockEchoConfig(raw string) (MockEchoConfig, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return defaultMockEchoConfig(), nil
	}
	if !strings.HasPrefix(raw, "{") {
		return MockEchoConfig{}, fmt.Errorf("mock-echo config must be JSON object")
	}
	var spec MockEchoSpec
	if err := json.Unmarshal([]byte(raw), &spec); err != nil {
		return MockEchoConfig{}, fmt.Errorf("invalid mock-echo config: %w", err)
	}
	cfg := defaultMockEchoConfig()
	if spec.Repeat != 0 {
		cfg.Repeat = spec.Repeat
	}
	if strings.TrimSpace(spec.Delay) != "" {
		d, err := time.ParseDuration(strings.TrimSpace(spec.Delay))
		if err != nil {
			return MockEchoConfig{}, fmt.Errorf("invalid mock-echo delay: %w", err)
		}
		cfg.Delay = d
	}
	if strings.TrimSpace(spec.Prefix) != "" {
		cfg.Prefix = spec.Prefix
	}
	return normalizeMockEchoConfig(cfg)
}

// ParseMockEchoModel parses the full mock-echo provider string.
func ParseMockEchoModel(modelWithProvider string) (MockEchoConfig, error) {
	raw := strings.TrimSpace(modelWithProvider)
	if raw == "" {
		return MockEchoConfig{}, fmt.Errorf("mock-echo model is empty")
	}
	if strings.EqualFold(raw, ProviderMockEcho) {
		return defaultMockEchoConfig(), nil
	}
	prefix := ProviderMockEcho + ":"
	if len(raw) < len(prefix) || !strings.EqualFold(raw[:len(prefix)], prefix) {
		return MockEchoConfig{}, fmt.Errorf("unsupported mock-echo model: %s", raw)
	}
	return ParseMockEchoConfig(raw[len(prefix):])
}

// MockEchoLLM deterministically echoes latest user text.
type MockEchoLLM struct {
	cfg MockEchoConfig
}

func NewMockEchoLLM() *MockEchoLLM {
	return NewMockEchoLLMWithConfig(defaultMockEchoConfig())
}

func NewMockEchoLLMWithConfig(cfg MockEchoConfig) *MockEchoLLM {
	normalized, err := normalizeMockEchoConfig(cfg)
	if err != nil {
		normalized = defaultMockEchoConfig()
	}
	return &MockEchoLLM{cfg: normalized}
}

func (m *MockEchoLLM) Name() string {
	return "mock-echo"
}

func (m *MockEchoLLM) GenerateContent(ctx context.Context, req *adkmodel.LLMRequest, stream bool) iter.Seq2[*adkmodel.LLMResponse, error] {
	cfg := defaultMockEchoConfig()
	if m != nil {
		cfg = m.cfg
	}
	if normalized, err := normalizeMockEchoConfig(cfg); err == nil {
		cfg = normalized
	}
	text := buildMockEchoText(latestUserText(req), cfg)

	return func(yield func(*adkmodel.LLMResponse, error) bool) {
		if err := sleepContextForMockEcho(ctx, cfg.Delay); err != nil {
			yield(nil, err)
			return
		}
		if stream {
			var b strings.Builder
			for _, r := range []rune(text) {
				if ctx.Err() != nil {
					yield(nil, ctx.Err())
					return
				}
				b.WriteRune(r)
				if !yield(&adkmodel.LLMResponse{
					Content: genai.NewContentFromText(b.String(), genai.RoleModel),
					Partial: true,
				}, nil) {
					return
				}
			}
		}
		yield(&adkmodel.LLMResponse{
			Content:      genai.NewContentFromText(text, genai.RoleModel),
			Partial:      false,
			TurnComplete: true,
		}, nil)
	}
}

func buildMockEchoText(userText string, cfg MockEchoConfig) string {
	txt := strings.TrimSpace(userText)
	if txt == "" {
		txt = "ok"
	}
	line := cfg.Prefix + txt
	if cfg.Repeat <= 1 {
		return line
	}
	lines := make([]string, 0, cfg.Repeat)
	for i := 0; i < cfg.Repeat; i++ {
		lines = append(lines, line)
	}
	return strings.Join(lines, "\n")
}

func sleepContextForMockEcho(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

func latestUserText(req *adkmodel.LLMRequest) string {
	if req == nil || len(req.Contents) == 0 {
		return ""
	}
	for i := len(req.Contents) - 1; i >= 0; i-- {
		c := req.Contents[i]
		if c == nil {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(c.Role), string(genai.RoleUser)) {
			continue
		}
		var parts []string
		for _, p := range c.Parts {
			if p == nil {
				continue
			}
			if t := strings.TrimSpace(p.Text); t != "" {
				parts = append(parts, t)
			}
		}
		if len(parts) > 0 {
			return strings.Join(parts, "\n")
		}
	}
	return ""
}

var _ adkmodel.LLM = (*MockEchoLLM)(nil)

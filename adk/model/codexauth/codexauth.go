package codexauth

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	AuthSourceEnvCodexAPIKey  = "env_codex_api_key"
	AuthSourceCodexKeychain   = "codex_keychain"
	AuthSourceCodexAuthFile   = "codex_auth_file"
	AuthSourceEnvOpenAIAPIKey = "env_openai_api_key"
	AuthSourceNone            = "none"

	ChatGPTCodexBaseURL = "https://chatgpt.com/backend-api/codex"
	OpenAIBaseURL       = "https://api.openai.com/v1"
)

type Resolution struct {
	Source    string
	APIKey    string
	AccountID string
	CodexHome string
}

type ResolveOptions struct {
	Env                map[string]string
	Platform           string
	CodexHome          string
	ReadFileText       func(path string) (string, error)
	PathExists         func(path string) bool
	RealPath           func(path string) (string, error)
	ReadKeychainSecret func(service, account string) (string, error)
}

func Resolve(opts ResolveOptions) Resolution {
	env := opts.Env
	if env == nil {
		env = currentEnvMap()
	}
	platform := strings.ToLower(strings.TrimSpace(opts.Platform))
	if platform == "" {
		platform = runtime.GOOS
	}
	codexHome := strings.TrimSpace(opts.CodexHome)
	if codexHome == "" {
		codexHome = resolveCodexHome(env)
	}
	readFileText := opts.ReadFileText
	if readFileText == nil {
		readFileText = func(path string) (string, error) {
			raw, err := os.ReadFile(path)
			if err != nil {
				return "", err
			}
			return string(raw), nil
		}
	}
	pathExists := opts.PathExists
	if pathExists == nil {
		pathExists = func(path string) bool {
			_, err := os.Stat(path)
			return err == nil
		}
	}
	realPath := opts.RealPath
	if realPath == nil {
		realPath = filepath.EvalSymlinks
	}
	readKeychainSecret := opts.ReadKeychainSecret
	if readKeychainSecret == nil {
		readKeychainSecret = readKeychainSecretDefault
	}

	if apiKey := strings.TrimSpace(env["CODEX_API_KEY"]); apiKey != "" {
		return Resolution{Source: AuthSourceEnvCodexAPIKey, APIKey: apiKey, AccountID: extractAccountID(apiKey), CodexHome: codexHome}
	}

	if platform == "darwin" {
		account := computeKeychainAccount(codexHome, realPath)
		if raw, err := readKeychainSecret("Codex Auth", account); err == nil {
			if apiKey := extractAPIKey(raw); apiKey != "" {
				return Resolution{Source: AuthSourceCodexKeychain, APIKey: apiKey, AccountID: extractAccountID(raw), CodexHome: codexHome}
			}
		}
	}

	authFile := filepath.Join(codexHome, "auth.json")
	if pathExists(authFile) {
		if raw, err := readFileText(authFile); err == nil {
			if apiKey := extractAPIKey(raw); apiKey != "" {
				return Resolution{Source: AuthSourceCodexAuthFile, APIKey: apiKey, AccountID: extractAccountID(raw), CodexHome: codexHome}
			}
		}
	}

	if apiKey := strings.TrimSpace(env["OPENAI_API_KEY"]); apiKey != "" {
		return Resolution{Source: AuthSourceEnvOpenAIAPIKey, APIKey: apiKey, AccountID: extractAccountID(apiKey), CodexHome: codexHome}
	}

	return Resolution{Source: AuthSourceNone, CodexHome: codexHome}
}

func ResolveBaseURL(source string, env map[string]string) string {
	if env == nil {
		env = currentEnvMap()
	}
	if explicit := firstNonEmpty(
		strings.TrimSpace(env["CODEX_BASE_URL"]),
		strings.TrimSpace(env["OPENAI_BASE_URL"]),
	); explicit != "" {
		return explicit
	}
	if source == AuthSourceCodexKeychain || source == AuthSourceCodexAuthFile {
		return ChatGPTCodexBaseURL
	}
	return OpenAIBaseURL
}

func ExtractAccountID(raw string) string {
	return extractAccountID(raw)
}

func currentEnvMap() map[string]string {
	out := map[string]string{}
	for _, entry := range os.Environ() {
		if idx := strings.Index(entry, "="); idx > 0 {
			out[entry[:idx]] = entry[idx+1:]
		}
	}
	return out
}

func resolveCodexHome(env map[string]string) string {
	if home := strings.TrimSpace(env["CODEX_HOME"]); home != "" {
		return home
	}
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return ".codex"
	}
	return filepath.Join(homeDir, ".codex")
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func extractAPIKey(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}

	var parsed struct {
		OpenAIAPIKey string `json:"OPENAI_API_KEY"`
		Tokens       *struct {
			AccessToken string `json:"access_token"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
		if apiKey := strings.TrimSpace(parsed.OpenAIAPIKey); apiKey != "" {
			return apiKey
		}
		if parsed.Tokens != nil {
			if token := strings.TrimSpace(parsed.Tokens.AccessToken); token != "" {
				return token
			}
		}
	}

	if !strings.Contains(trimmed, "{") && !strings.Contains(trimmed, "}") && !strings.ContainsAny(trimmed, " \n\r\t") {
		return trimmed
	}
	return ""
}

func extractAccountID(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}

	var parsed struct {
		Tokens *struct {
			AccessToken string `json:"access_token"`
		} `json:"tokens"`
	}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
		if parsed.Tokens != nil {
			if accountID := extractAccountIDFromToken(parsed.Tokens.AccessToken); accountID != "" {
				return accountID
			}
		}
	}
	return extractAccountIDFromToken(trimmed)
}

func extractAccountIDFromToken(token string) string {
	parts := strings.Split(strings.TrimSpace(token), ".")
	if len(parts) < 2 {
		return ""
	}
	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return ""
	}
	var payload struct {
		Auth *struct {
			ChatGPTAccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return ""
	}
	if payload.Auth == nil {
		return ""
	}
	return strings.TrimSpace(payload.Auth.ChatGPTAccountID)
}

func computeKeychainAccount(codexHome string, realPath func(path string) (string, error)) string {
	canonical := codexHome
	if resolved, err := realPath(codexHome); err == nil && strings.TrimSpace(resolved) != "" {
		canonical = resolved
	}
	sum := sha256.Sum256([]byte(canonical))
	return "cli|" + hex.EncodeToString(sum[:])[:16]
}

func readKeychainSecretDefault(service, account string) (string, error) {
	out, err := exec.Command("security", "find-generic-password", "-s", service, "-a", account, "-w").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(out)), nil
}

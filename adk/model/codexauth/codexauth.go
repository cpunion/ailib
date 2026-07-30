package codexauth

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const (
	AuthSourceEnvCodexAPIKey  = "env_codex_api_key"
	AuthSourceCodexKeychain   = "codex_keychain"
	AuthSourceCodexAuthFile   = "codex_auth_file"
	AuthSourceEnvOpenAIAPIKey = "env_openai_api_key"
	AuthSourceNone            = "none"

	ChatGPTCodexBaseURL = "https://chatgpt.com/backend-api/codex"
	OpenAIBaseURL       = "https://api.openai.com/v1"

	codexRefreshWindow = 5 * time.Minute
	appServerTimeout   = 30 * time.Second
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
	Now                func() time.Time
	RefreshCodexAuth   func(context.Context, string, bool) (Resolution, error)
}

var refreshMu sync.Mutex

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

// ResolveFresh resolves credentials and asks Codex's own AuthManager to refresh
// managed ChatGPT OAuth credentials when their JWT is close to expiry. Ailib
// deliberately does not exchange or persist Codex refresh tokens itself.
func ResolveFresh(ctx context.Context, opts ResolveOptions) (Resolution, error) {
	resolved := Resolve(opts)
	if !isManagedCodexOAuth(resolved.Source) ||
		!tokenExpiresSoon(resolved.APIKey, nowFromOptions(opts)) {
		return resolved, nil
	}

	refreshMu.Lock()
	defer refreshMu.Unlock()

	// Another request or Codex process may have refreshed the shared store while
	// this caller waited for the in-process guard.
	resolved = Resolve(opts)
	if !isManagedCodexOAuth(resolved.Source) ||
		!tokenExpiresSoon(resolved.APIKey, nowFromOptions(opts)) {
		return resolved, nil
	}
	return refreshWithCodex(ctx, opts, resolved, true)
}

// RefreshAfterUnauthorized refreshes the exact managed credential that an
// upstream request rejected. If Codex rotated its shared credential store
// after that request began, the newer non-expiring credential is used without
// forcing an unnecessary second rotation.
func RefreshAfterUnauthorized(
	ctx context.Context,
	opts ResolveOptions,
	failed Resolution,
) (Resolution, error) {
	if !isManagedCodexOAuth(failed.Source) {
		return Resolve(opts), nil
	}
	refreshMu.Lock()
	defer refreshMu.Unlock()

	reloaded := Resolve(opts)
	if reloaded.APIKey != "" &&
		reloaded.APIKey != failed.APIKey &&
		!tokenExpiresSoon(reloaded.APIKey, nowFromOptions(opts)) {
		return reloaded, nil
	}
	if strings.TrimSpace(reloaded.APIKey) == "" {
		reloaded = failed
	}
	if !isManagedCodexOAuth(reloaded.Source) {
		return reloaded, nil
	}
	return refreshWithCodex(ctx, opts, reloaded, true)
}

func refreshWithCodex(
	ctx context.Context,
	opts ResolveOptions,
	current Resolution,
	force bool,
) (Resolution, error) {
	refresh := opts.RefreshCodexAuth
	if refresh == nil {
		refresh = refreshViaCodexAppServer
	}
	refreshed, err := refresh(ctx, current.CodexHome, force)
	if err != nil {
		return Resolution{}, fmt.Errorf("refresh Codex credentials: %w", err)
	}
	if strings.TrimSpace(refreshed.APIKey) == "" {
		return Resolution{}, fmt.Errorf("refresh Codex credentials: Codex returned no access token")
	}
	if force && refreshed.APIKey == current.APIKey {
		return Resolution{}, fmt.Errorf(
			"refresh Codex credentials: Codex returned the unchanged access token",
		)
	}
	if tokenExpiresSoon(refreshed.APIKey, nowFromOptions(opts)) {
		return Resolution{}, fmt.Errorf(
			"refresh Codex credentials: Codex returned an access token that is already expiring",
		)
	}
	if strings.TrimSpace(refreshed.Source) == "" {
		refreshed.Source = current.Source
	}
	if strings.TrimSpace(refreshed.CodexHome) == "" {
		refreshed.CodexHome = current.CodexHome
	}
	if strings.TrimSpace(refreshed.AccountID) == "" {
		refreshed.AccountID = extractAccountID(refreshed.APIKey)
	}
	return refreshed, nil
}

func isManagedCodexOAuth(source string) bool {
	return source == AuthSourceCodexKeychain ||
		source == AuthSourceCodexAuthFile
}

func nowFromOptions(opts ResolveOptions) time.Time {
	if opts.Now != nil {
		return opts.Now()
	}
	return time.Now()
}

func tokenExpiresSoon(token string, now time.Time) bool {
	expiresAt, ok := tokenExpiry(token)
	if !ok {
		return false
	}
	return !expiresAt.After(now.Add(codexRefreshWindow))
}

func tokenExpiry(token string) (time.Time, bool) {
	parts := strings.Split(strings.TrimSpace(token), ".")
	if len(parts) < 2 {
		return time.Time{}, false
	}
	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return time.Time{}, false
	}
	var payload struct {
		ExpiresAt int64 `json:"exp"`
	}
	if err := json.Unmarshal(payloadBytes, &payload); err != nil ||
		payload.ExpiresAt <= 0 {
		return time.Time{}, false
	}
	return time.Unix(payload.ExpiresAt, 0), true
}

func refreshViaCodexAppServer(
	ctx context.Context,
	codexHome string,
	force bool,
) (Resolution, error) {
	ctx, cancel := context.WithTimeout(ctx, appServerTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "codex", "app-server", "--stdio")
	if strings.TrimSpace(codexHome) != "" {
		cmd.Env = replaceEnvironmentValue(
			os.Environ(),
			"CODEX_HOME",
			codexHome,
		)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return Resolution{}, fmt.Errorf("open Codex stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return Resolution{}, fmt.Errorf("open Codex stdout: %w", err)
	}
	var stderr synchronizedBoundedBuffer
	cmd.Stderr = &stderr
	if err := cmd.Start(); err != nil {
		return Resolution{}, fmt.Errorf("start Codex app-server: %w", err)
	}
	defer func() {
		_ = stdin.Close()
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		_ = cmd.Wait()
	}()

	encoder := json.NewEncoder(stdin)
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 0, 64*1024), 4<<20)
	if err := encoder.Encode(map[string]any{
		"id":     1,
		"method": "initialize",
		"params": map[string]any{
			"clientInfo": map[string]any{
				"name":    "ailib-codex-responses-proxy",
				"version": "1",
			},
			"capabilities": map[string]any{
				"experimentalApi": true,
			},
		},
	}); err != nil {
		return Resolution{}, fmt.Errorf("initialize Codex app-server: %w", err)
	}
	if _, err := readCodexResponse(scanner, 1); err != nil {
		return Resolution{}, withCodexStderr(err, stderr.String())
	}
	if err := encoder.Encode(map[string]any{
		"method": "initialized",
	}); err != nil {
		return Resolution{}, fmt.Errorf("acknowledge Codex initialization: %w", err)
	}
	if err := encoder.Encode(codexAuthRefreshRequest(force)); err != nil {
		return Resolution{}, fmt.Errorf("request Codex auth status: %w", err)
	}
	_, err = readCodexResponse(scanner, 2)
	if err != nil {
		return Resolution{}, withCodexStderr(err, stderr.String())
	}
	resolved := Resolve(ResolveOptions{CodexHome: codexHome})
	if strings.TrimSpace(resolved.APIKey) == "" {
		return Resolution{}, fmt.Errorf(
			"Codex refreshed auth but its credential store returned no token",
		)
	}
	return resolved, nil
}

func codexAuthRefreshRequest(force bool) map[string]any {
	return map[string]any{
		"id":     2,
		"method": "account/read",
		"params": map[string]any{
			"refreshToken": force,
		},
	}
}

func readCodexResponse(
	scanner *bufio.Scanner,
	wantID int,
) (json.RawMessage, error) {
	for scanner.Scan() {
		var envelope struct {
			ID     json.RawMessage `json:"id"`
			Result json.RawMessage `json:"result"`
			Error  *struct {
				Code    int    `json:"code"`
				Message string `json:"message"`
			} `json:"error"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &envelope); err != nil {
			continue
		}
		if strings.TrimSpace(string(envelope.ID)) != fmt.Sprint(wantID) {
			continue
		}
		if envelope.Error != nil {
			return nil, fmt.Errorf(
				"Codex app-server error %d: %s",
				envelope.Error.Code,
				envelope.Error.Message,
			)
		}
		if len(envelope.Result) == 0 {
			return nil, fmt.Errorf("Codex app-server response %d has no result", wantID)
		}
		return envelope.Result, nil
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read Codex app-server response: %w", err)
	}
	return nil, io.ErrUnexpectedEOF
}

func withCodexStderr(err error, stderr string) error {
	detail := strings.TrimSpace(stderr)
	if detail == "" {
		return err
	}
	const maxDetail = 2048
	if len(detail) > maxDetail {
		detail = detail[:maxDetail]
	}
	return fmt.Errorf("%w: %s", err, detail)
}

func replaceEnvironmentValue(
	environment []string,
	key string,
	value string,
) []string {
	prefix := key + "="
	out := make([]string, 0, len(environment)+1)
	for _, entry := range environment {
		if !strings.HasPrefix(entry, prefix) {
			out = append(out, entry)
		}
	}
	return append(out, prefix+value)
}

type synchronizedBoundedBuffer struct {
	mu   sync.Mutex
	data []byte
}

func (b *synchronizedBoundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	const limit = 2048
	remaining := limit - len(b.data)
	if remaining > 0 {
		if len(p) < remaining {
			remaining = len(p)
		}
		b.data = append(b.data, p[:remaining]...)
	}
	return len(p), nil
}

func (b *synchronizedBoundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(append([]byte(nil), b.data...))
}

func ResolveBaseURL(source string, env map[string]string) string {
	if env == nil {
		env = currentEnvMap()
	}
	if source == AuthSourceCodexKeychain || source == AuthSourceCodexAuthFile {
		if strings.TrimSpace(env["AILIB_CODEX_ALLOW_UNSAFE_BASE_URL"]) == "1" {
			if explicit := strings.TrimSpace(env["CODEX_BASE_URL"]); explicit != "" {
				return explicit
			}
		}
		return ChatGPTCodexBaseURL
	}
	if source == AuthSourceEnvCodexAPIKey {
		if explicit := strings.TrimSpace(env["CODEX_BASE_URL"]); explicit != "" {
			return explicit
		}
		return OpenAIBaseURL
	}
	if explicit := strings.TrimSpace(env["OPENAI_BASE_URL"]); explicit != "" {
		return explicit
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

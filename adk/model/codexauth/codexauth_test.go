package codexauth

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestResolveBaseURLDoesNotRedirectCodexOAuthCredentials(t *testing.T) {
	env := map[string]string{
		"OPENAI_BASE_URL": "https://attacker.invalid/v1",
		"CODEX_BASE_URL":  "https://proxy.invalid/codex",
	}
	for _, source := range []string{
		AuthSourceCodexKeychain,
		AuthSourceCodexAuthFile,
	} {
		if got := ResolveBaseURL(source, env); got != ChatGPTCodexBaseURL {
			t.Fatalf("source=%s baseURL=%q", source, got)
		}
	}
}

func TestResolveBaseURLRequiresExplicitUnsafeOptInForCodexOAuth(t *testing.T) {
	env := map[string]string{
		"CODEX_BASE_URL":                    "https://proxy.invalid/codex",
		"AILIB_CODEX_ALLOW_UNSAFE_BASE_URL": "1",
	}
	if got := ResolveBaseURL(AuthSourceCodexAuthFile, env); got != env["CODEX_BASE_URL"] {
		t.Fatalf("baseURL=%q", got)
	}
}

func TestResolveBaseURLKeepsOpenAIAPIKeyConfiguration(t *testing.T) {
	env := map[string]string{
		"OPENAI_BASE_URL": "https://openai-compatible.invalid/v1",
	}
	if got := ResolveBaseURL(AuthSourceEnvOpenAIAPIKey, env); got != env["OPENAI_BASE_URL"] {
		t.Fatalf("baseURL=%q", got)
	}
}

func TestResolveFreshUsesCodexAuthorityNearExpiry(t *testing.T) {
	now := time.Date(2026, 7, 30, 12, 0, 0, 0, time.UTC)
	current := testCodexJWT(t, now.Add(4*time.Minute), "acct-old")
	fresh := testCodexJWT(t, now.Add(time.Hour), "acct-new")
	refreshCount := 0

	resolved, err := ResolveFresh(context.Background(), ResolveOptions{
		Env:       map[string]string{},
		Platform:  "linux",
		CodexHome: "/codex-home",
		PathExists: func(path string) bool {
			return path == "/codex-home/auth.json"
		},
		ReadFileText: func(string) (string, error) {
			raw, marshalErr := json.Marshal(map[string]any{
				"tokens": map[string]any{
					"access_token": current,
				},
			})
			return string(raw), marshalErr
		},
		Now: func() time.Time { return now },
		RefreshCodexAuth: func(
			context.Context,
			string,
			bool,
		) (Resolution, error) {
			refreshCount++
			return Resolution{APIKey: fresh}, nil
		},
	})
	if err != nil {
		t.Fatalf("ResolveFresh: %v", err)
	}
	if refreshCount != 1 ||
		resolved.Source != AuthSourceCodexAuthFile ||
		resolved.APIKey != fresh ||
		resolved.AccountID != "acct-new" {
		t.Fatalf("refreshCount=%d resolved=%+v", refreshCount, resolved)
	}
}

func TestResolveFreshSkipsValidAndUnmanagedCredentials(t *testing.T) {
	now := time.Date(2026, 7, 30, 12, 0, 0, 0, time.UTC)
	valid := testCodexJWT(t, now.Add(time.Hour), "acct-valid")
	for _, tc := range []struct {
		name string
		opts ResolveOptions
	}{
		{
			name: "managed_valid",
			opts: ResolveOptions{
				Env:       map[string]string{},
				Platform:  "linux",
				CodexHome: "/codex-home",
				PathExists: func(path string) bool {
					return path == "/codex-home/auth.json"
				},
				ReadFileText: func(string) (string, error) {
					raw, err := json.Marshal(map[string]any{
						"tokens": map[string]any{"access_token": valid},
					})
					return string(raw), err
				},
			},
		},
		{
			name: "env_api_key",
			opts: ResolveOptions{
				Env: map[string]string{"CODEX_API_KEY": "sk-static"},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			refreshCount := 0
			tc.opts.Now = func() time.Time { return now }
			tc.opts.RefreshCodexAuth = func(
				context.Context,
				string,
				bool,
			) (Resolution, error) {
				refreshCount++
				return Resolution{}, nil
			}
			if _, err := ResolveFresh(context.Background(), tc.opts); err != nil {
				t.Fatalf("ResolveFresh: %v", err)
			}
			if refreshCount != 0 {
				t.Fatalf("refreshCount=%d", refreshCount)
			}
		})
	}
}

func TestResolveFreshRejectsUnchangedOrExpiringRefreshResult(t *testing.T) {
	now := time.Date(2026, 7, 30, 12, 0, 0, 0, time.UTC)
	current := testCodexJWT(t, now.Add(4*time.Minute), "acct")
	for _, test := range []struct {
		name      string
		refreshed string
		wantError string
	}{
		{
			name:      "unchanged",
			refreshed: current,
			wantError: "unchanged access token",
		},
		{
			name:      "still expiring",
			refreshed: testCodexJWT(t, now.Add(3*time.Minute), "acct"),
			wantError: "already expiring",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			_, err := ResolveFresh(context.Background(), ResolveOptions{
				Env:       map[string]string{},
				Platform:  "linux",
				CodexHome: "/codex-home",
				PathExists: func(path string) bool {
					return path == "/codex-home/auth.json"
				},
				ReadFileText: func(string) (string, error) {
					raw, marshalErr := json.Marshal(map[string]any{
						"tokens": map[string]any{
							"access_token": current,
						},
					})
					return string(raw), marshalErr
				},
				Now: func() time.Time { return now },
				RefreshCodexAuth: func(
					context.Context,
					string,
					bool,
				) (Resolution, error) {
					return Resolution{APIKey: test.refreshed}, nil
				},
			})
			if err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("error=%v want containing %q", err, test.wantError)
			}
		})
	}
}

func TestRefreshAfterUnauthorizedUsesExternallyRotatedCredential(t *testing.T) {
	now := time.Date(2026, 7, 30, 12, 0, 0, 0, time.UTC)
	failed := testCodexJWT(t, now.Add(time.Hour), "acct-old")
	rotated := testCodexJWT(t, now.Add(2*time.Hour), "acct-new")
	refreshCount := 0

	resolved, err := RefreshAfterUnauthorized(
		context.Background(),
		ResolveOptions{
			Env:       map[string]string{},
			Platform:  "linux",
			CodexHome: "/codex-home",
			PathExists: func(path string) bool {
				return path == "/codex-home/auth.json"
			},
			ReadFileText: func(string) (string, error) {
				raw, marshalErr := json.Marshal(map[string]any{
					"tokens": map[string]any{
						"access_token": rotated,
					},
				})
				return string(raw), marshalErr
			},
			Now: func() time.Time { return now },
			RefreshCodexAuth: func(
				context.Context,
				string,
				bool,
			) (Resolution, error) {
				refreshCount++
				return Resolution{}, nil
			},
		},
		Resolution{
			Source:    AuthSourceCodexAuthFile,
			APIKey:    failed,
			CodexHome: "/codex-home",
		},
	)
	if err != nil {
		t.Fatalf("RefreshAfterUnauthorized: %v", err)
	}
	if refreshCount != 0 ||
		resolved.APIKey != rotated ||
		resolved.AccountID != "acct-new" {
		t.Fatalf("refreshCount=%d resolved=%+v", refreshCount, resolved)
	}
}

func TestCodexRefreshUsesProviderIndependentAccountRead(t *testing.T) {
	request := codexAuthRefreshRequest(true)
	if request["method"] != "account/read" {
		t.Fatalf("request=%#v", request)
	}
	params, _ := request["params"].(map[string]any)
	if params["refreshToken"] != true {
		t.Fatalf("params=%#v", params)
	}
}

func TestReadCodexResponseIgnoresNotifications(t *testing.T) {
	input := "{\"method\":\"account/updated\"}\n" +
		"{\"id\":2,\"result\":{\"authToken\":\"fresh\"}}\n"
	result, err := readCodexResponse(
		bufio.NewScanner(strings.NewReader(input)),
		2,
	)
	if err != nil {
		t.Fatalf("readCodexResponse: %v", err)
	}
	if string(result) != `{"authToken":"fresh"}` {
		t.Fatalf("result=%s", result)
	}
}

func TestCodexAppServerAuthIntegration(t *testing.T) {
	if os.Getenv("AILIB_CODEX_LIVE_TEST") != "1" {
		t.Skip("set AILIB_CODEX_LIVE_TEST=1 to use local Codex auth")
	}
	resolved, err := refreshViaCodexAppServer(
		context.Background(),
		resolveCodexHome(currentEnvMap()),
		false,
	)
	if err != nil {
		t.Fatalf("refreshViaCodexAppServer: %v", err)
	}
	if resolved.APIKey == "" || resolved.AccountID == "" {
		t.Fatalf(
			"missing live Codex credential metadata: token=%t account=%q",
			resolved.APIKey != "",
			resolved.AccountID,
		)
	}
}

func TestSynchronizedBoundedBufferConcurrentAccess(t *testing.T) {
	var buffer synchronizedBoundedBuffer
	var writers sync.WaitGroup
	for range 8 {
		writers.Add(1)
		go func() {
			defer writers.Done()
			for range 100 {
				_, _ = buffer.Write([]byte("stderr detail\n"))
				_ = buffer.String()
			}
		}()
	}
	writers.Wait()
	if got := len(buffer.String()); got > 2048 {
		t.Fatalf("buffer length=%d", got)
	}
}

func testCodexJWT(t *testing.T, expiresAt time.Time, accountID string) string {
	t.Helper()
	payload, err := json.Marshal(map[string]any{
		"exp": expiresAt.Unix(),
		"https://api.openai.com/auth": map[string]any{
			"chatgpt_account_id": accountID,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	return "header." +
		base64.RawURLEncoding.EncodeToString(payload) +
		".signature"
}

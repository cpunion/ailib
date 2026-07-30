package main

import (
	"bufio"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cpunion/ailib/adk/model/codexauth"
)

func TestCodexProxyForwardsResponsesWithoutChangingTheStream(t *testing.T) {
	const stream = "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"opaque\"}}\n\n"
	upstream := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		r *http.Request,
	) {
		if r.URL.Path != "/responses" {
			t.Fatalf("upstream path=%q", r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer codex-token" ||
			r.Header.Get("chatgpt-account-id") != "acct-1" ||
			r.Header.Get("OpenAI-Beta") != "responses=experimental" ||
			r.Header.Get("originator") != "ailib" {
			t.Fatalf("upstream headers=%v", r.Header)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatal(err)
		}
		if string(body) != `{"model":"gpt-5.6-sol","stream":true}` {
			t.Fatalf("upstream body=%s", body)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, stream)
	}))
	defer upstream.Close()

	proxy := httptest.NewServer(&codexProxy{
		baseURL:         upstream.URL,
		capabilityToken: strings.Repeat("c", 32),
		resolveAuth: func(context.Context) (codexauth.Resolution, error) {
			return codexauth.Resolution{
				Source: codexauth.AuthSourceCodexAuthFile,
				APIKey: "codex-token", AccountID: "acct-1",
			}, nil
		},
		client: upstream.Client(),
	})
	defer proxy.Close()
	request, err := http.NewRequestWithContext(
		context.Background(),
		http.MethodPost,
		proxy.URL+"/v1/responses",
		strings.NewReader(`{"model":"gpt-5.6-sol","stream":true}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set(
		"Authorization",
		"Bearer "+strings.Repeat("c", 32),
	)
	response, err := proxy.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	body, err := io.ReadAll(bufio.NewReader(response.Body))
	if err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != http.StatusOK ||
		response.Header.Get("Content-Type") != "text/event-stream" ||
		string(body) != stream {
		t.Fatalf(
			"status=%d content-type=%q body=%q",
			response.StatusCode,
			response.Header.Get("Content-Type"),
			body,
		)
	}
}

func TestCodexProxyRejectsMissingCapability(t *testing.T) {
	recorder := httptest.NewRecorder()
	(&codexProxy{
		capabilityToken: strings.Repeat("c", 32),
	}).ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodPost, "/v1/responses", nil),
	)
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("status=%d body=%s", recorder.Code, recorder.Body)
	}
}

func TestCodexProxyRefreshesCredentialsPerRequest(t *testing.T) {
	var upstreamAuth []string
	upstream := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		r *http.Request,
	) {
		upstreamAuth = append(upstreamAuth, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	resolveCount := 0
	proxy := httptest.NewServer(&codexProxy{
		baseURL:         upstream.URL,
		capabilityToken: strings.Repeat("c", 32),
		resolveAuth: func(context.Context) (codexauth.Resolution, error) {
			resolveCount++
			return codexauth.Resolution{
				Source:    codexauth.AuthSourceCodexAuthFile,
				APIKey:    "rotated-" + string(rune('0'+resolveCount)),
				AccountID: "acct-1",
			}, nil
		},
		client: upstream.Client(),
	})
	defer proxy.Close()

	for range 2 {
		request, err := http.NewRequest(
			http.MethodPost,
			proxy.URL+"/v1/responses",
			strings.NewReader(`{"stream":true}`),
		)
		if err != nil {
			t.Fatal(err)
		}
		request.Header.Set(
			"Authorization",
			"Bearer "+strings.Repeat("c", 32),
		)
		response, err := proxy.Client().Do(request)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, response.Body)
		response.Body.Close()
	}
	if resolveCount != 2 ||
		len(upstreamAuth) != 2 ||
		upstreamAuth[0] != "Bearer rotated-1" ||
		upstreamAuth[1] != "Bearer rotated-2" {
		t.Fatalf(
			"resolveCount=%d upstreamAuth=%v",
			resolveCount,
			upstreamAuth,
		)
	}
}

func TestCodexProxyRefreshesAndRetriesOnceOnUnauthorized(t *testing.T) {
	var upstreamAuth []string
	upstream := httptest.NewServer(http.HandlerFunc(func(
		w http.ResponseWriter,
		r *http.Request,
	) {
		upstreamAuth = append(upstreamAuth, r.Header.Get("Authorization"))
		if len(upstreamAuth) == 1 {
			http.Error(w, "expired", http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer upstream.Close()

	refreshCount := 0
	proxy := httptest.NewServer(&codexProxy{
		baseURL:         upstream.URL,
		capabilityToken: strings.Repeat("c", 32),
		resolveAuth: func(context.Context) (codexauth.Resolution, error) {
			return codexauth.Resolution{
				Source: codexauth.AuthSourceCodexAuthFile,
				APIKey: "expired-token",
			}, nil
		},
		forceRefresh: func(
			_ context.Context,
			failed codexauth.Resolution,
		) (codexauth.Resolution, error) {
			if failed.APIKey != "expired-token" {
				t.Fatalf("failed credential=%+v", failed)
			}
			refreshCount++
			return codexauth.Resolution{
				Source: codexauth.AuthSourceCodexAuthFile,
				APIKey: "fresh-token",
			}, nil
		},
		client: upstream.Client(),
	})
	defer proxy.Close()

	request, err := http.NewRequest(
		http.MethodPost,
		proxy.URL+"/v1/responses",
		strings.NewReader(`{"stream":true}`),
	)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set(
		"Authorization",
		"Bearer "+strings.Repeat("c", 32),
	)
	response, err := proxy.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK ||
		refreshCount != 1 ||
		len(upstreamAuth) != 2 ||
		upstreamAuth[0] != "Bearer expired-token" ||
		upstreamAuth[1] != "Bearer fresh-token" {
		t.Fatalf(
			"status=%d refreshCount=%d upstreamAuth=%v",
			response.StatusCode,
			refreshCount,
			upstreamAuth,
		)
	}
}

func TestCodexProxyListenAddressMustBeLoopback(t *testing.T) {
	for _, address := range []string{
		"127.0.0.1:18777",
		"[::1]:18777",
	} {
		if err := validateListenAddress(address); err != nil {
			t.Fatalf("address %q: %v", address, err)
		}
	}
	for _, address := range []string{
		":18777",
		"0.0.0.0:18777",
		"192.168.1.20:18777",
		"localhost:18777",
	} {
		if err := validateListenAddress(address); err == nil {
			t.Fatalf("address %q should be rejected", address)
		}
	}
}

func TestCodexProxyRejectsNonResponsesPaths(t *testing.T) {
	recorder := httptest.NewRecorder()
	(&codexProxy{}).ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "/health", nil),
	)
	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status=%d", recorder.Code)
	}
}

func TestCodexProxyReportsItsSourceRevision(t *testing.T) {
	recorder := httptest.NewRecorder()
	(&codexProxy{
		revision: "ailib-revision",
		modified: true,
	}).ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "/healthz", nil),
	)
	if recorder.Code != http.StatusOK ||
		recorder.Header().Get("Content-Type") != "application/json" ||
		!strings.Contains(recorder.Body.String(), `"revision":"ailib-revision"`) ||
		!strings.Contains(recorder.Body.String(), `"modified":true`) {
		t.Fatalf(
			"status=%d headers=%v body=%s",
			recorder.Code,
			recorder.Header(),
			recorder.Body,
		)
	}
}

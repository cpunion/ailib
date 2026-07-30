package main

import (
	"bytes"
	"context"
	"crypto/subtle"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
	"time"

	"github.com/cpunion/ailib/adk/model/codexauth"
)

const defaultListenAddress = "127.0.0.1:18777"
const minimumCapabilityTokenBytes = 32

type codexProxy struct {
	baseURL         string
	capabilityToken string
	resolveAuth     func(context.Context) (codexauth.Resolution, error)
	forceRefresh    func(
		context.Context,
		codexauth.Resolution,
	) (codexauth.Resolution, error)
	client   *http.Client
	revision string
	modified bool
}

func main() {
	resolveAuth := func(ctx context.Context) (codexauth.Resolution, error) {
		return codexauth.ResolveFresh(ctx, codexauth.ResolveOptions{})
	}
	auth, err := resolveAuth(context.Background())
	if err != nil {
		log.Fatalf("codex-responses-proxy: %v", err)
	}
	if strings.TrimSpace(auth.APIKey) == "" {
		log.Fatal("codex-responses-proxy: no Codex/OpenAI credentials found")
	}
	capabilityToken := strings.TrimSpace(
		os.Getenv("AILIB_CODEX_PROXY_TOKEN"),
	)
	if len(capabilityToken) < minimumCapabilityTokenBytes {
		log.Fatalf(
			"codex-responses-proxy: AILIB_CODEX_PROXY_TOKEN must be at least %d bytes",
			minimumCapabilityTokenBytes,
		)
	}
	proxy := &codexProxy{
		capabilityToken: capabilityToken,
		resolveAuth:     resolveAuth,
		forceRefresh: func(
			ctx context.Context,
			failed codexauth.Resolution,
		) (codexauth.Resolution, error) {
			return codexauth.RefreshAfterUnauthorized(
				ctx,
				codexauth.ResolveOptions{},
				failed,
			)
		},
		client: &http.Client{
			Timeout: 15 * time.Minute,
		},
	}
	proxy.revision, proxy.modified = sourceRevision()
	addr := strings.TrimSpace(os.Getenv("AILIB_CODEX_PROXY_ADDR"))
	if addr == "" {
		addr = defaultListenAddress
	}
	if err := validateListenAddress(addr); err != nil {
		log.Fatalf("codex-responses-proxy: %v", err)
	}
	server := &http.Server{
		Addr:              addr,
		Handler:           proxy,
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Printf(
		"codex-responses-proxy listening on http://%s/v1 (auth=%s)",
		addr,
		auth.Source,
	)
	log.Fatal(server.ListenAndServe())
}

func (p *codexProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet && r.URL.Path == "/healthz" {
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(
			w,
			`{"ok":true,"module":"github.com/cpunion/ailib","revision":%q,"modified":%t}`,
			p.revision,
			p.modified,
		)
		return
	}
	if r.Method != http.MethodPost ||
		(r.URL.Path != "/v1/responses" && r.URL.Path != "/responses") {
		http.NotFound(w, r)
		return
	}
	if !p.authorized(r) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return
	}
	resolveAuth := p.resolveAuth
	if resolveAuth == nil {
		resolveAuth = func(ctx context.Context) (codexauth.Resolution, error) {
			return codexauth.ResolveFresh(ctx, codexauth.ResolveOptions{})
		}
	}
	auth, err := resolveAuth(r.Context())
	if err != nil {
		http.Error(w, err.Error(), http.StatusServiceUnavailable)
		return
	}
	if strings.TrimSpace(auth.APIKey) == "" {
		http.Error(
			w,
			"Codex credentials unavailable",
			http.StatusServiceUnavailable,
		)
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "read request body", http.StatusBadRequest)
		return
	}
	response, err := p.sendUpstream(r, auth, body)
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		http.Error(w, "codex upstream request failed", http.StatusBadGateway)
		return
	}
	if response.StatusCode == http.StatusUnauthorized &&
		isManagedCodexSource(auth.Source) {
		response.Body.Close()
		forceRefresh := p.forceRefresh
		if forceRefresh == nil {
			forceRefresh = func(
				ctx context.Context,
				failed codexauth.Resolution,
			) (codexauth.Resolution, error) {
				return codexauth.RefreshAfterUnauthorized(
					ctx,
					codexauth.ResolveOptions{},
					failed,
				)
			}
		}
		auth, err = forceRefresh(r.Context(), auth)
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		response, err = p.sendUpstream(r, auth, body)
		if err != nil {
			if r.Context().Err() != nil {
				return
			}
			http.Error(w, "codex upstream retry failed", http.StatusBadGateway)
			return
		}
	}
	defer response.Body.Close()
	for key, values := range response.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	w.WriteHeader(response.StatusCode)
	if err := copyStreaming(r.Context(), w, response.Body); err != nil &&
		r.Context().Err() == nil {
		log.Printf("codex-responses-proxy: stream copy failed: %v", err)
	}
}

func (p *codexProxy) sendUpstream(
	r *http.Request,
	auth codexauth.Resolution,
	body []byte,
) (*http.Response, error) {
	baseURL := strings.TrimSpace(p.baseURL)
	if baseURL == "" {
		baseURL = codexauth.ResolveBaseURL(auth.Source, nil)
	}
	target := strings.TrimRight(baseURL, "/") + "/responses"
	request, err := http.NewRequestWithContext(
		r.Context(),
		http.MethodPost,
		target,
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("create upstream request: %w", err)
	}
	copyRequestHeader(request.Header, r.Header, "Content-Type")
	copyRequestHeader(request.Header, r.Header, "Accept")
	copyRequestHeader(request.Header, r.Header, "session-id")
	copyRequestHeader(request.Header, r.Header, "x-client-request-id")
	request.Header.Set("Authorization", "Bearer "+auth.APIKey)
	if strings.TrimSpace(auth.AccountID) != "" {
		request.Header.Set("chatgpt-account-id", auth.AccountID)
	}
	request.Header.Set("OpenAI-Beta", "responses=experimental")
	request.Header.Set("originator", "ailib")

	client := p.client
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	return response, nil
}

func isManagedCodexSource(source string) bool {
	return source == codexauth.AuthSourceCodexAuthFile ||
		source == codexauth.AuthSourceCodexKeychain
}

func (p *codexProxy) authorized(r *http.Request) bool {
	if p == nil || r == nil {
		return false
	}
	const prefix = "Bearer "
	header := strings.TrimSpace(r.Header.Get("Authorization"))
	if !strings.HasPrefix(header, prefix) {
		return false
	}
	provided := strings.TrimSpace(strings.TrimPrefix(header, prefix))
	expected := strings.TrimSpace(p.capabilityToken)
	if len(expected) < minimumCapabilityTokenBytes ||
		len(provided) != len(expected) {
		return false
	}
	return subtle.ConstantTimeCompare(
		[]byte(provided),
		[]byte(expected),
	) == 1
}

func validateListenAddress(address string) error {
	host, rawPort, err := net.SplitHostPort(strings.TrimSpace(address))
	if err != nil {
		return fmt.Errorf("invalid listen address %q: %w", address, err)
	}
	ip := net.ParseIP(strings.TrimSpace(host))
	if ip == nil || !ip.IsLoopback() {
		return fmt.Errorf(
			"listen address %q must use a numeric loopback IP",
			address,
		)
	}
	port, err := strconv.Atoi(rawPort)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("listen address %q has invalid port", address)
	}
	return nil
}

func sourceRevision() (revision string, modified bool) {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown", false
	}
	for _, setting := range info.Settings {
		switch setting.Key {
		case "vcs.revision":
			revision = setting.Value
		case "vcs.modified":
			modified = setting.Value == "true"
		}
	}
	if strings.TrimSpace(revision) == "" {
		revision = "unknown"
	}
	return revision, modified
}

func copyRequestHeader(dst, src http.Header, key string) {
	for _, value := range src.Values(key) {
		dst.Add(key, value)
	}
}

func copyStreaming(ctx context.Context, dst http.ResponseWriter, src io.Reader) error {
	buffer := make([]byte, 32<<10)
	flusher, _ := dst.(http.Flusher)
	for {
		n, err := src.Read(buffer)
		if n > 0 {
			if _, writeErr := dst.Write(buffer[:n]); writeErr != nil {
				return writeErr
			}
			if flusher != nil {
				flusher.Flush()
			}
		}
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return err
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("stream canceled: %w", ctx.Err())
		default:
		}
	}
}

# ailib

Go libraries extracted for reuse.

## Codex Responses proxy

`cmd/codex-responses-proxy` exposes the locally authenticated Codex Responses
transport to a loopback client without copying the Codex credential into that
client:

```bash
export AILIB_CODEX_PROXY_TOKEN="$(openssl rand -hex 32)"
go run ./cmd/codex-responses-proxy
```

The proxy only accepts a numeric loopback listen address and requires the same
capability token as a bearer token on `/v1/responses`. It resolves Codex
credentials for every request, asks Codex's own app-server to refresh managed
OAuth credentials near expiry, and performs one guarded refresh/retry after an
upstream `401`. Credential rotation therefore does not require a proxy restart,
and `ailib` never reads or persists the Codex refresh token itself.
Keychain/auth-file credentials always use the official Codex origin; a custom
`CODEX_BASE_URL` requires the explicit
`AILIB_CODEX_ALLOW_UNSAFE_BASE_URL=1` opt-in.

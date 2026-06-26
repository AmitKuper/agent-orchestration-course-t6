"""Ollama-compatible shim that forwards LLM calls to OpenRouter.

The read-only submodule's ``Gatekeeper`` speaks only Anthropic or Ollama-native
``POST {OLLAMA_BASE_URL}/api/chat`` (no auth header, reply read from
``data["message"]["content"]``). OpenRouter is OpenAI-format with Bearer auth,
so it cannot be plugged in directly. This tiny stdlib server *looks like Ollama*
to the Gatekeeper and proxies each request to OpenRouter — letting us use free
cloud models for the cosmetic NL message without modifying the submodule.

Run it, then point the submodule at it::

    uv run python scripts/openrouter_adapter.py
    # in the submodule .env:
    OLLAMA_BASE_URL=http://localhost:11500
    LLM_MODEL=meta-llama/llama-3.2-3b-instruct:free   # any OpenRouter model id

Config (env): OPENROUTER_API_KEY (required), OPENROUTER_BASE_URL
(default https://openrouter.ai/api/v1), ADAPTER_PORT (default 11500 — distinct
from Ollama's 11434 so both backends can coexist).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_DEFAULT_PORT = 11500
_TIMEOUT = 120.0


def _forward_to_openrouter(payload: dict) -> dict:
    """Translate an Ollama chat request, call OpenRouter, return Ollama shape.

    Args:
        payload: Ollama-style body ``{"model", "messages", "stream"}``.

    Returns:
        Ollama-style reply ``{"message": {...}, "prompt_eval_count", ...}``.

    Raises:
        RuntimeError: If ``OPENROUTER_API_KEY`` is not set.
        urllib.error.HTTPError: If OpenRouter returns a non-2xx response.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    base = os.environ.get("OPENROUTER_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
    body = json.dumps({
        "model": payload.get("model"),
        "messages": payload.get("messages", []),
        "stream": False,
        "max_tokens": payload.get("max_tokens", 512),
    }).encode()
    request = urllib.request.Request(
        base + "/chat/completions", data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/evya1/AI-actor-game",
            "X-Title": "Cop & Thief actor_t6",
        },
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as resp:  # noqa: S310
        data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": usage.get("prompt_tokens", 0),
        "eval_count": usage.get("completion_tokens", 0),
    }


class _Handler(BaseHTTPRequestHandler):
    """Serves the single ``POST /api/chat`` route the Gatekeeper invokes."""

    def log_message(self, *_args: object) -> None:
        """Silence default per-request stderr logging."""

    def _send(self, code: int, obj: dict) -> None:
        """Write a JSON response with the given status code."""
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802  (BaseHTTPRequestHandler API)
        """Handle ``/api/chat`` by proxying to OpenRouter."""
        if self.path.rstrip("/") != "/api/chat":
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            self._send(200, _forward_to_openrouter(payload))
        except urllib.error.HTTPError as exc:
            self._send(exc.code, {"error": exc.read().decode(errors="replace")})
        except (urllib.error.URLError, RuntimeError, KeyError) as exc:
            self._send(502, {"error": str(exc)})


def main() -> None:
    """Start the adapter server on ADAPTER_PORT until interrupted."""
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    port = int(os.environ.get("ADAPTER_PORT", _DEFAULT_PORT))
    server = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    model_hint = os.environ.get("LLM_MODEL", "<set LLM_MODEL to an OpenRouter id>")
    print(f"[openrouter-adapter] listening on http://localhost:{port}/api/chat")  # noqa: T201
    print(f"[openrouter-adapter] point OLLAMA_BASE_URL at it; LLM_MODEL={model_hint}")  # noqa: T201
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()

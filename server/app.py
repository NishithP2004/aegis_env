# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Aegis Env Environment.

This module creates an HTTP server that exposes the AegisEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - Custom UI:  GET /web, GET /web/benchmark (served from server/web)
    - Benchmark: GET /api/benchmark/models, POST /api/benchmark/run
    - OpenEnv HTTP/WS API: same app mounted at / and /openenv (explicit routes win first)
        - POST /reset, /openenv/reset
        - POST /step, /openenv/step
        - GET  /state, /openenv/state
        - GET  /schema, /openenv/schema
        - WS   /ws, /openenv/ws

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import AegisAction, AegisObservation
    from .aegis_env_environment import AegisEnvironment
    from .benchmark import fetch_model_ids, run_single_model_episode
except ImportError:
    from models import AegisAction, AegisObservation
    from server.aegis_env_environment import AegisEnvironment
    from server.benchmark import fetch_model_ids, run_single_model_episode


# Create the OpenEnv app with default UI + HTTP/WS API.
openenv_app = create_app(
    AegisEnvironment,
    AegisAction,
    AegisObservation,
    env_name="aegis-env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)


@openenv_app.exception_handler(Exception)
async def _openenv_uncaught_exception_handler(_request, exc: Exception) -> JSONResponse:
    # Surface the actual exception in HTTP responses to speed up local debugging.
    # (Uvicorn still logs the full traceback; this keeps the frontend informative.)
    return JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {exc!s}"},
    )

# Top-level FastAPI app (serves custom UI + mounts OpenEnv app).
app = FastAPI(title="AEGIS-Env")
app.mount("/openenv", openenv_app)

WEB_DIR = Path(__file__).resolve().parent / "web"

# Register explicit /web routes BEFORE mounting StaticFiles at /web, otherwise the
# mount swallows /web/* and FileResponse handlers for /web/benchmark never run.


@app.get("/", include_in_schema=False)
def _root() -> RedirectResponse:
    return RedirectResponse(url="/web")


@app.get("/web", include_in_schema=False)
def _web_index() -> FileResponse:
    return FileResponse(str(WEB_DIR / "index.html"))


@app.get("/web/benchmark", include_in_schema=False)
def _web_benchmark() -> FileResponse:
    return FileResponse(str(WEB_DIR / "benchmark.html"))


@app.get("/web/benchmark.html", include_in_schema=False)
def _web_benchmark_html() -> FileResponse:
    return FileResponse(str(WEB_DIR / "benchmark.html"))


app.mount("/web", StaticFiles(directory=str(WEB_DIR), html=True), name="web")


# --- Simple stateful HTTP API (used by custom /web UI) ---
#
# The OpenEnv WebSocket client is the canonical interface, but the mounted OpenEnv
# HTTP step endpoint may not coerce dict actions into the Pydantic Action model
# in some versions. These endpoints provide a minimal, predictable JSON API for
# the frontend: reset → step → … → step.
_http_env = AegisEnvironment()


class EnvResetRequest(BaseModel):
    seed: int | None = None
    episode_id: str | None = None
    task_name: str | None = None


class EnvStepRequest(BaseModel):
    action: dict[str, Any]
    timeout_s: float | None = None


@app.post("/api/env/reset")
def api_env_reset(req: EnvResetRequest) -> dict[str, Any]:
    obs = _http_env.reset(seed=req.seed, episode_id=req.episode_id, task_name=req.task_name)
    return {"observation": obs.model_dump(), "reward": getattr(obs, "reward", None), "done": bool(getattr(obs, "done", False))}


@app.post("/api/env/step")
def api_env_step(req: EnvStepRequest) -> dict[str, Any]:
    obs = _http_env.step(req.action, timeout_s=req.timeout_s)
    return {"observation": obs.model_dump(), "reward": getattr(obs, "reward", None), "done": bool(getattr(obs, "done", False))}


# --- Model benchmark (Ollama / OpenAI-compatible): list models + run 5-model comparison ---
_benchmark_env = AegisEnvironment()


@app.get("/api/benchmark/models")
def api_benchmark_models(api_root: str = "https://ollama.com/v1") -> dict[str, Any]:
    try:
        ids = fetch_model_ids(api_root)
        return {"models": ids, "api_root": api_root.strip().rstrip("/")}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


class BenchmarkRunRequest(BaseModel):
    """Run the same episode for each model; only `model` differs in chat completions."""

    models: List[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="1–5 distinct model ids (e.g. Ollama model names)",
    )
    task_name: str = Field(pattern="^(easy|medium|hard)$")
    max_steps: int = Field(default=10, ge=1, le=200)
    seed: int | None = Field(
        default=None,
        description="Optional RNG seed for reset (same seed for every model in the run)",
    )
    api_base_url: str = Field(
        default="https://ollama.com/v1",
        description="OpenAI-compatible chat base URL (same for all models)",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional; if empty, uses HF_TOKEN/API_KEY/OPENAI_API_KEY or the literal 'ollama'",
    )

    @field_validator("models")
    @classmethod
    def _unique_models(cls, v: List[str]) -> List[str]:
        cleaned = [m.strip() for m in v if isinstance(m, str) and m.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("models must be unique")
        if len(cleaned) < 1:
            raise ValueError("at least one model is required")
        return cleaned


@app.post("/api/benchmark/run")
def api_benchmark_run(req: BenchmarkRunRequest) -> dict[str, Any]:
    key = (req.api_key or "").strip() or _llm_api_key() or "ollama"
    base = req.api_base_url.strip().rstrip("/")
    client = OpenAI(base_url=base, api_key=key)

    results: List[dict[str, Any]] = []
    for model in req.models:
        results.append(
            run_single_model_episode(
                _benchmark_env,
                client,
                model,
                req.task_name,
                req.max_steps,
                req.seed,
            )
        )

    return {
        "task_name": req.task_name,
        "seed": req.seed,
        "max_steps_cap": req.max_steps,
        "api_base_url": base,
        "results": results,
    }


class ChatMessage(BaseModel):
    role: str
    content: str


class LLMCompleteRequest(BaseModel):
    """OpenAI-compatible chat completion for the web UI auto-run (same env as inference.py)."""

    messages: List[ChatMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128_000)


def _llm_api_key() -> str:
    # Match inference.py _get_api_key(): HF_TOKEN or API_KEY (plus common OpenAI name).
    return str(
        os.environ.get("HF_TOKEN")
        or os.environ.get("API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()


def _llm_base_url() -> str:
    return os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"


def _llm_model_name() -> str:
    return os.getenv("MODEL_NAME") or "meta-llama/Llama-3.2-3B-Instruct"


@app.post("/api/llm/complete")
def llm_complete(req: LLMCompleteRequest) -> dict[str, Any]:
    """
    Proxy chat completion for browser auto-run. Uses HF_TOKEN, API_KEY, or OPENAI_API_KEY,
    plus API_BASE_URL and MODEL_NAME (same contract as inference.py).
    """
    api_key = _llm_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail=(
                "LLM not configured: set HF_TOKEN, API_KEY, or OPENAI_API_KEY "
                "in the server environment for auto-run."
            ),
        )

    client = OpenAI(base_url=_llm_base_url(), api_key=api_key)
    model = _llm_model_name()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [m.model_dump() for m in req.messages],
        "temperature": req.temperature,
    }
    if req.max_tokens > 0:
        payload["max_tokens"] = req.max_tokens

    try:
        completion = client.chat.completions.create(**payload)
    except Exception as first_exc:
        try:
            fb: dict[str, Any] = {
                "model": model,
                "messages": [m.model_dump() for m in req.messages],
                "temperature": req.temperature,
            }
            if req.max_tokens > 0:
                fb["max_completion_tokens"] = req.max_tokens
            completion = client.chat.completions.create(**fb)
        except Exception as second_exc:
            raise HTTPException(
                status_code=502,
                detail=f"LLM request failed: {first_exc!s}; fallback: {second_exc!s}",
            ) from second_exc

    if not completion.choices:
        raise HTTPException(status_code=502, detail="LLM returned no choices")

    msg = completion.choices[0].message
    content = getattr(msg, "content", None)
    text: str
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
            else:
                t = getattr(part, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        text = "\n".join(parts).strip()
    else:
        text = ""

    return {"content": text, "model": model}


# Stock OpenEnv paths (/reset, /state, /step, …) — registered after /, /web, /api/* so those win.
app.mount("/", openenv_app)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m aegis_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn aegis_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.port == 8000:
        main()
    else:
        main(port=args.port)

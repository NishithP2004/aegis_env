# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark helpers: list OpenAI-compatible `/v1/models`, run episodes per model."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from inference import (
        _action_log_str,
        _get_action_with_retry,
        _one_line,
        build_user_prompt,
    )
    from models import AegisObservation
except ImportError:  # pragma: no cover — allow `python -m server.app` from package subdir
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[1]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from inference import (
        _action_log_str,
        _get_action_with_retry,
        _one_line,
        build_user_prompt,
    )
    from models import AegisObservation

TEMPERATURE = 0.2
MAX_TOKENS = 4096


def fetch_model_ids(api_root: str, timeout_s: float = 45.0) -> List[str]:
    """
    GET {api_root}/models — OpenAI-compatible listing (Ollama exposes this at /v1/models).
    """
    root = api_root.strip().rstrip("/")
    url = root if root.endswith("/models") else f"{root}/models"
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "aegis-env-benchmark/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        raise RuntimeError(f"HTTP {e.code} listing models from {url}: {body or e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to reach {url}: {e!s}") from e

    ids: List[str] = []
    for item in payload.get("data") or []:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("name")
            if isinstance(mid, str) and mid.strip():
                ids.append(mid.strip())
    # Native Ollama `/api/tags` shape (optional fallback)
    if not ids:
        for item in payload.get("models") or []:
            if isinstance(item, dict):
                mid = item.get("name") or item.get("model")
                if isinstance(mid, str) and mid.strip():
                    ids.append(mid.strip())
    return sorted(set(ids))


def run_single_model_episode(
    env: Any,
    llm: OpenAI,
    model: str,
    task_name: str,
    max_steps: int,
    episode_seed: Optional[int],
) -> Dict[str, Any]:
    """
    One grading episode: only `model` changes vs other runs (same env instance, reset between models).
    """
    rewards: List[float] = []
    history: List[str] = []
    last_action: Optional[str] = None
    last_reward = 0.0
    obs: AegisObservation
    try:
        obs = env.reset(seed=episode_seed, task_name=task_name)
        for step in range(1, max_steps + 1):
            prompt = build_user_prompt(step, last_action, last_reward, history, obs)
            try:
                action, _text = _get_action_with_retry(
                    llm,
                    model,
                    prompt,
                    TEMPERATURE,
                    MAX_TOKENS,
                    float(obs.max_score) if obs.max_score else 1.0,
                    llm_enabled=True,
                )
            except Exception as e:
                rewards.append(0.0)
                history.append(f"step={step} parse_error={_one_line(str(e))}")
                last_action = None
                last_reward = 0.0
                continue

            out = env.step(action)
            r = float(getattr(out, "reward", None) or 0.0)
            rewards.append(r)
            last_action = _action_log_str(action)
            last_reward = r
            history.append(f"step={step} action={last_action} reward={r:.2f}")
            obs = out
            if bool(getattr(out, "done", False)):
                break

        return {
            "model": model,
            "rewards": rewards,
            "total_reward": float(sum(rewards)),
            "steps": len(rewards),
            "final_done": bool(getattr(obs, "done", False)),
            "error": None,
        }
    except Exception as e:
        return {
            "model": model,
            "rewards": rewards,
            "total_reward": float(sum(rewards)),
            "steps": len(rewards),
            "final_done": False,
            "error": f"{type(e).__name__}: {e}",
        }

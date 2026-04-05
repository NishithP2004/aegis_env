#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Baseline inference loop: OpenAI-compatible LLM API + AEGIS-Env (WebSocket client or local env).

Required environment variables:
  HF_TOKEN — Hugging Face API token (mandatory).

Environment variables with defaults:
  API_BASE_URL — LLM API base URL (default: Hugging Face OpenAI-compatible router).
  MODEL_NAME — Model identifier for chat completions.

Logging format (hackathon):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import List, Optional

from openai import OpenAI

from models import AegisAction, AegisObservation

# Defaults when API_BASE_URL / MODEL_NAME are unset (Hugging Face Inference + router).
_DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
_DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"


def _llm_api_base_url() -> str:
    return os.environ.get("API_BASE_URL", _DEFAULT_API_BASE_URL)


def _model_name() -> str:
    return os.environ.get("MODEL_NAME", _DEFAULT_MODEL_NAME)


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token is None or not str(token).strip():
        print(
            "HF_TOKEN is required (set your Hugging Face API token in the environment).",
            file=sys.stderr,
        )
        sys.exit(1)
    return str(token).strip()


def generate_grading_prompt(obs: AegisObservation) -> str:
    return f"""You are an expert Automated Grading Agent. Your goal is to evaluate a student's answer by combining the rigor of a strict evaluator with the constructive guidance of a mentor.

Follow this internal evaluation process:
1. Scrutinize: Conduct a detailed, criterion-by-criterion verification of the student's answer against the rubric.
2. Validate: Ensure your evaluation is fair, unbiased, and strictly adheres to the scoring limits.
3. Mentor: Translate your technical findings into clear, personalized, and actionable feedback that fosters a growth mindset.

You must output ONLY a raw JSON object. Do not wrap it in markdown formatting (e.g., no ```json). The JSON must have exactly these keys:
{{
  "final_score": <number strictly between 0 and {obs.max_score} inclusive>,
  "score_justification": "<string: detailed reasoning for the score based on the rubric>",
  "improvement_advice": "<string: specific, actionable recommendations for the student>"
}}

--- DATA ---
Question:
{obs.question}

Rubric:
{obs.rubric}

Maximum Score: {obs.max_score}

Student Answer:
{obs.student_answer}
"""


def _episode_seed() -> Optional[int]:
    raw = os.environ.get("EPISODE_SEED")
    if raw is None or raw == "":
        return None
    return int(raw)


def _parse_action(raw: str, max_score: float) -> AegisAction:
    data = json.loads(raw)
    return AegisAction(
        final_score=float(data["final_score"]),
        score_justification=str(data.get("score_justification", "")),
        improvement_advice=str(data.get("improvement_advice", "")),
    )


def _action_log_str(action: AegisAction) -> str:
    return json.dumps(
        {
            "final_score": action.final_score,
            "score_justification": action.score_justification[:200],
            "improvement_advice": action.improvement_advice[:200],
        },
        ensure_ascii=False,
    )


def _complete_grading(
    llm: OpenAI,
    model: str,
    prompt: str,
) -> str:
    completion = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()


def _strip_markdown_json_fence(text: str) -> str:
    if text.startswith("```"):
        lines = text.split("\n")
        return "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text


def run_local_episode(
    task_name: str,
    benchmark: str,
    model: str,
    llm: OpenAI,
) -> tuple[bool, int, List[float]]:
    """Run one episode using in-process ``AegisEnvironment`` (no HTTP server)."""
    from server.aegis_env_environment import AegisEnvironment

    rewards: List[float] = []
    steps = 0

    print(f"[START] task={task_name} env={benchmark} model={model}", flush=True)

    env = AegisEnvironment()
    try:
        obs = env.reset(seed=_episode_seed())
        prompt = generate_grading_prompt(obs)
        text = _strip_markdown_json_fence(_complete_grading(llm, model, prompt))

        try:
            action = _parse_action(text, obs.max_score)
        except Exception as e:
            err = f"parse_error: {e!s}"
            print(
                f"[STEP] step={steps + 1} action={text[:300]!r} reward={0.0:.2f} "
                f"done={True} error={err}",
                flush=True,
            )
            rewards.append(0.0)
            return False, steps + 1, rewards

        steps += 1
        out = env.step(action)
        r = float(out.reward or 0.0)
        rewards.append(r)
        print(
            f"[STEP] step={steps} action={_action_log_str(action)!r} reward={r:.2f} "
            f"done={out.done} error=null",
            flush=True,
        )
        return True, steps, rewards
    except Exception as e:
        err = f"{type(e).__name__}: {e!s}"
        print(
            f"[STEP] step={steps + 1} action=<none> reward={0.0:.2f} done={True} error={err}",
            flush=True,
        )
        traceback.print_exc()
        rewards.append(0.0)
        return False, steps + 1, rewards
    finally:
        env.close()


def run_client_episode(
    task_name: str,
    benchmark: str,
    model: str,
    openenv_base_url: str,
    llm: OpenAI,
) -> tuple[bool, int, List[float]]:
    """Run one episode via OpenEnv WebSocket client (server must be running)."""
    from client import AegisEnv

    rewards: List[float] = []
    steps = 0

    print(f"[START] task={task_name} env={benchmark} model={model}", flush=True)

    sync = AegisEnv(base_url=openenv_base_url).sync()
    try:
        with sync:
            r0 = sync.reset(seed=_episode_seed())
            obs = r0.observation
            prompt = generate_grading_prompt(obs)
            text = _strip_markdown_json_fence(_complete_grading(llm, model, prompt))

            try:
                action = _parse_action(text, obs.max_score)
            except Exception as e:
                err = f"parse_error: {e!s}"
                print(
                    f"[STEP] step={steps + 1} action={text[:300]!r} reward={0.0:.2f} "
                    f"done={True} error={err}",
                    flush=True,
                )
                rewards.append(0.0)
                return False, steps + 1, rewards

            steps += 1
            r1 = sync.step(action)
            r = float(r1.reward or 0.0)
            rewards.append(r)
            print(
                f"[STEP] step={steps} action={_action_log_str(action)!r} reward={r:.2f} "
                f"done={r1.done} error=null",
                flush=True,
            )
            return True, steps, rewards
    except Exception as e:
        success = False
        err = f"{type(e).__name__}: {e!s}"
        print(
            f"[STEP] step={steps + 1} action=<none> reward={0.0:.2f} done={True} error={err}",
            flush=True,
        )
        traceback.print_exc()
        if not rewards:
            rewards.append(0.0)
        return success, max(steps, 1), rewards


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="AEGIS-Env baseline inference (OpenAI-compatible LLM + HF_TOKEN)",
    )
    p.add_argument("--task-name", default="grading", help="Logged task name")
    p.add_argument("--benchmark", default="AEGIS-Env", help="Logged benchmark / env name")
    p.add_argument(
        "--model",
        default=None,
        help=f"Override MODEL_NAME (default from env or {_DEFAULT_MODEL_NAME!r})",
    )
    p.add_argument(
        "--api-base-url",
        default=None,
        dest="api_base_url",
        help=f"Override API_BASE_URL (default from env or {_DEFAULT_API_BASE_URL!r})",
    )
    p.add_argument(
        "--openenv-base-url",
        default=os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000"),
        help="OpenEnv server URL (ignored with --local)",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="Use in-process AegisEnvironment (requires MONGO_URI); no server",
    )
    args = p.parse_args(argv)

    hf_token = _require_hf_token()
    api_base = args.api_base_url if args.api_base_url is not None else _llm_api_base_url()
    model = args.model if args.model is not None else _model_name()

    llm = OpenAI(base_url=api_base, api_key=hf_token)

    if args.local:
        ok, n, rewards = run_local_episode(args.task_name, args.benchmark, model, llm)
    else:
        ok, n, rewards = run_client_episode(
            args.task_name,
            args.benchmark,
            model,
            args.openenv_base_url,
            llm,
        )

    rlist = ",".join(f"{x:.4f}" for x in rewards)
    print(f"[END] success={str(ok).lower()} steps={n} rewards={rlist}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

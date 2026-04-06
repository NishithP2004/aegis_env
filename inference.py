#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Hackathon inference script for AEGIS-Env.

MANDATORY / configurable environment variables:
  API_BASE_URL     LLM API endpoint (default set below).
  MODEL_NAME       Model id for chat completions (default set below).
  HF_TOKEN         Hugging Face / API key for the OpenAI client.
  LOCAL_IMAGE_NAME or IMAGE_NAME
                   If set, the environment client is created with from_docker_image().
  OPENENV_BASE_URL Used when neither --local nor a Docker image is set.

Stdout (exact line shapes):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from typing import List, Optional, Tuple

from openai import OpenAI
from openenv.core.client_types import StepResult

from models import AegisAction, AegisObservation

# --- Defaults (API_BASE_URL / MODEL_NAME only; HF_TOKEN has no default) ---
_DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
_DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.2-3B-Instruct"

API_BASE_URL = _DEFAULT_API_BASE_URL
MODEL_NAME = _DEFAULT_MODEL_NAME

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or ""
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK") or os.getenv("AEGIS_BENCHMARK") or "AEGIS-Env"
MAX_STEPS = int(os.getenv("MAX_STEPS", "1"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD", "0.1"))


def _llm_api_base_url(cli_override: Optional[str]) -> str:
    if cli_override is not None:
        return cli_override
    return os.getenv("API_BASE_URL") or API_BASE_URL


def _model_name(cli_override: Optional[str]) -> str:
    if cli_override is not None:
        return cli_override
    return os.getenv("MODEL_NAME") or MODEL_NAME


def _docker_image_name(cli_override: Optional[str]) -> str:
    if cli_override:
        return cli_override
    return (os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME") or "").strip()


def _done_str(done: bool) -> str:
    return str(done).lower()


def _episode_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.0
    s = sum(rewards) / len(rewards)
    return min(1.0, max(0.0, s))


def _one_line(s: str) -> str:
    return s.replace("\n", " ").replace("\r", " ")


def _last_action_error_from_result(sr: Optional[StepResult[AegisObservation]]) -> Optional[str]:
    if sr is None:
        return None
    md = getattr(sr.observation, "metadata", None) or {}
    return md.get("last_action_error") or md.get("error")


def _error_column(parse_exc: Optional[str], sr: Optional[StepResult[AegisObservation]]) -> str:
    if parse_exc is not None:
        return _one_line(parse_exc)
    lae = _last_action_error_from_result(sr)
    if lae is not None and str(lae).strip() != "":
        return _one_line(str(lae))
    return "null"


def _require_hf_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
    if token is None or not str(token).strip():
        print(
            "HF_TOKEN (or API_KEY) is required for the OpenAI client.",
            file=sys.stderr,
        )
        sys.exit(1)
    return str(token).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error_column: str,
) -> None:
    # Two spaces after [STEP] per hackathon format
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} "
        f"done={_done_str(done)} error={error_column}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


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


def _complete_grading(llm: OpenAI, model: str, prompt: str) -> str:
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
    }
    if MAX_TOKENS > 0:
        kwargs["max_tokens"] = MAX_TOKENS
    completion = llm.chat.completions.create(**kwargs)
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
) -> Tuple[bool, int, List[float]]:
    """In-process environment (sync)."""
    from server.aegis_env_environment import AegisEnvironment

    rewards: List[float] = []
    steps_taken = 0
    log_start(task_name, benchmark, model)

    env = AegisEnvironment()
    obs: AegisObservation
    try:
        obs = env.reset(seed=_episode_seed(), task_name=task_name)
        for step in range(1, MAX_STEPS + 1):
            if obs.done and steps_taken > 0:
                break

            prompt = generate_grading_prompt(obs)
            text = _strip_markdown_json_fence(_complete_grading(llm, model, prompt))

            try:
                action = _parse_action(text, obs.max_score)
            except Exception as e:
                err = f"parse_error: {e!s}"
                steps_taken = step
                rewards.append(0.0)
                log_step(step, repr(text[:300]), 0.0, True, _error_column(err, None))
                return False, steps_taken, rewards

            out = env.step(action)
            r = float(getattr(out, "reward", None) or 0.0)
            rewards.append(r)
            steps_taken = step
            log_step(
                step,
                repr(_action_log_str(action)),
                r,
                bool(getattr(out, "done", False)),
                _error_column(None, None),
            )
            obs = out
            if getattr(out, "done", False):
                break

        return True, steps_taken, rewards
    except Exception as e:
        err = f"{type(e).__name__}: {e!s}"
        steps_taken = max(steps_taken, 1)
        rewards.append(0.0)
        log_step(steps_taken, "<none>", 0.0, True, _error_column(err, None))
        traceback.print_exc()
        return False, steps_taken, rewards
    finally:
        env.close()


async def run_remote_episode_async(
    task_name: str,
    benchmark: str,
    model: str,
    llm: OpenAI,
    openenv_base_url: str,
    docker_image: str,
) -> Tuple[bool, int, List[float]]:
    """WebSocket client: optional Docker image or HTTP base URL."""
    from client import AegisEnv

    rewards: List[float] = []
    steps_taken = 0
    log_start(task_name, benchmark, model)

    env: Optional["AegisEnv"] = None
    try:
        if docker_image:
            # Ensure the container sees the selected task.
            env = await AegisEnv.from_docker_image(
                docker_image,
                env_vars={"TASK_NAME": task_name},
            )
        else:
            env = AegisEnv(base_url=openenv_base_url)
            await env.connect()

        result = await env.reset(seed=_episode_seed(), task_name=task_name)

        for step in range(1, MAX_STEPS + 1):
            if result.done and steps_taken > 0:
                break

            obs = result.observation
            prompt = generate_grading_prompt(obs)
            text = _strip_markdown_json_fence(_complete_grading(llm, model, prompt))

            try:
                action = _parse_action(text, obs.max_score)
            except Exception as e:
                err = f"parse_error: {e!s}"
                steps_taken = step
                rewards.append(0.0)
                log_step(step, repr(text[:300]), 0.0, True, _error_column(err, None))
                return False, steps_taken, rewards

            result = await env.step(action)
            r = float(result.reward or 0.0)
            rewards.append(r)
            steps_taken = step
            log_step(
                step,
                repr(_action_log_str(action)),
                r,
                result.done,
                _error_column(None, result),
            )
            if result.done:
                break

        return True, steps_taken, rewards
    except Exception as e:
        err = f"{type(e).__name__}: {e!s}"
        steps_taken = max(steps_taken, 1)
        if not rewards:
            rewards.append(0.0)
        log_step(steps_taken, "<none>", 0.0, True, _error_column(err, None))
        traceback.print_exc()
        return False, steps_taken, rewards
    finally:
        if env is not None:
            await env.close()


async def run_episode_async(
    args: argparse.Namespace,
    model: str,
    llm: OpenAI,
) -> Tuple[bool, int, List[float]]:
    docker_image = _docker_image_name(getattr(args, "docker_image", None) or None)
    if args.local:
        return await asyncio.to_thread(
            run_local_episode,
            args.task_name,
            args.benchmark,
            model,
            llm,
        )
    return await run_remote_episode_async(
        args.task_name,
        args.benchmark,
        model,
        llm,
        args.openenv_base_url,
        docker_image,
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="AEGIS-Env inference (OpenAI client + async OpenEnv)",
    )
    p.add_argument(
        "--task-name",
        default=TASK_NAME,
        help="Logged task (env TASK_NAME / AEGIS_TASK)",
    )
    p.add_argument(
        "--benchmark",
        default=BENCHMARK,
        help="Logged benchmark (env BENCHMARK / AEGIS_BENCHMARK)",
    )
    p.add_argument("--model", default=None, help="Override MODEL_NAME")
    p.add_argument("--api-base-url", default=None, dest="api_base_url")
    p.add_argument(
        "--openenv-base-url",
        default=os.environ.get("OPENENV_BASE_URL", "http://127.0.0.1:8000"),
    )
    p.add_argument(
        "--docker-image",
        default=None,
        help="Override LOCAL_IMAGE_NAME / IMAGE_NAME for from_docker_image()",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="In-process environment (MONGO_URI); no HTTP/Docker client",
    )
    args = p.parse_args(argv)

    success = False
    steps_out = 0
    rewards_out: List[float] = []
    exit_code = 1

    try:
        hf_token = _require_hf_token()
        api_base = _llm_api_base_url(args.api_base_url)
        model = _model_name(args.model)
        llm = OpenAI(base_url=api_base, api_key=hf_token)

        _, steps_out, rewards_out = asyncio.run(run_episode_async(args, model, llm))
        score = _episode_score(rewards_out)
        success = score >= SUCCESS_SCORE_THRESHOLD
        exit_code = 0 if success else 1
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        success = False
        exit_code = 1
    finally:
        log_end(
            success,
            steps_out,
            _episode_score(rewards_out),
            rewards_out,
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

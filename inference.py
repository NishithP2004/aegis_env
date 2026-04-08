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
  LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                   method.
  OPENENV_BASE_URL The OpenEnv HTTP/WebSocket base URL (used when LOCAL_IMAGE_NAME is not set and --local is not used).

Stdout (exact line shapes):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from typing import List, Optional, Tuple

from openai import OpenAI
from openenv.core.client_types import StepResult

from models import AegisAction, AegisObservation

# --- Defaults (API_BASE_URL / MODEL_NAME only; HF_TOKEN has no default) ---
_DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
_DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.2-3B-Instruct"

API_BASE_URL = _DEFAULT_API_BASE_URL
MODEL_NAME = _DEFAULT_MODEL_NAME

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or ""

OPENENV_BASE_URL = os.getenv("OPENENV_BASE_URL") or "http://127.0.0.1:8000"
TASK_NAME = os.getenv("TASK_NAME", "easy")
BENCHMARK = os.getenv("BENCHMARK", "AEGIS-Env")
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
    return (LOCAL_IMAGE_NAME or "").strip()


def _done_str(done: bool) -> str:
    return str(done).lower()


def _episode_score(rewards: List[float]) -> float:
    if not rewards:
        # Must be strictly between 0 and 1, even after formatting to 2 decimals.
        return 0.01
    s = sum(rewards) / len(rewards)
    s = min(1.0, max(0.0, s))
    # Clamp to an exclusive range so we never emit 0.00 or 1.00 in logs.
    if s <= 0.0:
        return 0.01
    if s >= 1.0:
        return 0.99
    # Avoid rounding to 0.00 / 1.00 when printed with 2 decimals.
    if s < 0.01:
        return 0.01
    if s > 0.99:
        return 0.99
    return s


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


def _get_api_key() -> str:
    # Validator environments may omit API keys. We must never crash; in that case
    # we will fall back to a deterministic dummy action in the LLM call path.
    return str(os.environ.get("HF_TOKEN") or os.environ.get("API_KEY") or "").strip()


def _fallback_action_json() -> str:
    # Minimal valid action JSON that will always parse.
    return json.dumps(
        {
            "final_score": 0.0,
            "score_justification": "",
            "improvement_advice": "",
        },
        ensure_ascii=False,
    )


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


def build_user_prompt(
    step: int,
    last_action: Optional[str],
    last_reward: float,
    history: List[str],
    obs: AegisObservation,
) -> str:
    # Keep everything single-line friendly; the LLM can still read newlines,
    # but we avoid embedding uncontrolled exceptions/newlines in log fields.
    hist = "\n".join(history[-6:]) if history else "None"
    last_action_s = last_action if last_action is not None else "None"
    return (
        f"Step: {step}\n"
        f"Last action: {last_action_s}\n"
        f"Last reward: {last_reward:.2f}\n"
        f"Previous steps:\n{hist}\n\n"
        f"{generate_grading_prompt(obs)}"
    )


def _episode_seed(args: argparse.Namespace) -> Optional[int]:
    raw = getattr(args, "episode_seed", None)
    if raw is None:
        raw = os.environ.get("EPISODE_SEED")
    if raw is None or str(raw) == "":
        return None
    return int(raw)


def _parse_action(raw: str, max_score: float) -> AegisAction:
    decoder = json.JSONDecoder()
    cleaned = _strip_markdown_json_fence(raw).strip()
    data = None
    first_err: Optional[Exception] = None

    # Tolerate reasoning/thinking prefixes by scanning for the first JSON object.
    for i, ch in enumerate(cleaned):
        if ch != "{":
            continue
        try:
            candidate, _ = decoder.raw_decode(cleaned[i:])
            if isinstance(candidate, dict):
                data = candidate
                break
        except Exception as e:
            if first_err is None:
                first_err = e

    if data is None:
        if first_err is not None:
            raise first_err
        raise ValueError("No JSON object found in model output")

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
    temperature: float,
    max_tokens: int,
    llm_enabled: bool,
) -> str:
    if not llm_enabled:
        return _fallback_action_json()

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    fallback_kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if max_tokens > 0:
        kwargs["max_tokens"] = max_tokens
        fallback_kwargs["max_completion_tokens"] = max_tokens

    # Some OpenAI-compatible providers only accept one token-limit field.
    try:
        completion = llm.chat.completions.create(**kwargs)
    except Exception as first_exc:
        try:
            completion = llm.chat.completions.create(**fallback_kwargs)
        except Exception:
            return _fallback_action_json()

    if not completion.choices:
        return _fallback_action_json()
    msg = completion.choices[0].message
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Some providers/modes return structured content parts.
        text_parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    text_parts.append(t)
            else:
                t = getattr(part, "text", None)
                if isinstance(t, str):
                    text_parts.append(t)
        return "\n".join(text_parts).strip()
    return ""


def _repair_prompt_from_invalid_output(raw: str, max_score: float) -> str:
    clipped = raw[:2000]
    return (
        "Your previous response was not valid JSON.\n"
        "Return ONLY a valid JSON object (no markdown, no extra text) with exactly these keys:\n"
        '{\n'
        f'  "final_score": <number between 0 and {max_score}>,\n'
        '  "score_justification": "<string>",\n'
        '  "improvement_advice": "<string>"\n'
        "}\n\n"
        "Invalid response to fix:\n"
        f"{clipped}"
    )


def _get_action_with_retry(
    llm: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_score: float,
    llm_enabled: bool,
) -> Tuple[AegisAction, str]:
    text = _strip_markdown_json_fence(
        _complete_grading(llm, model, prompt, temperature, max_tokens, llm_enabled)
    )
    try:
        return _parse_action(text, max_score), text
    except Exception as first_exc:
        repair_prompt = _repair_prompt_from_invalid_output(text, max_score)
        repaired = _strip_markdown_json_fence(
            _complete_grading(llm, model, repair_prompt, 0.0, max_tokens, llm_enabled)
        )
        try:
            return _parse_action(repaired, max_score), repaired
        except Exception as second_exc:
            raise ValueError(
                f"parse_error_after_retry: first={first_exc!s}; second={second_exc!s}"
            ) from second_exc


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
    max_steps: int,
    temperature: float,
    max_tokens: int,
    episode_seed: Optional[int],
    success_score_threshold: float,
) -> Tuple[bool, int, List[float]]:
    """In-process environment (sync)."""
    from server.aegis_env_environment import AegisEnvironment

    rewards: List[float] = []
    steps_taken = 0
    history: List[str] = []
    last_action: Optional[str] = None
    last_reward = 0.0
    log_start(task_name, benchmark, model)

    env = AegisEnvironment()
    obs: AegisObservation
    try:
        for step in range(1, max_steps + 1):
            # AEGIS-Env can be a single-step episode (done after one step).
            # We keep running new episodes (reset -> step) until success threshold
            # is met or max_steps is exhausted.
            obs = env.reset(seed=episode_seed, task_name=task_name)

            prompt = build_user_prompt(step, last_action, last_reward, history, obs)
            text = ""
            try:
                action, text = _get_action_with_retry(
                    llm,
                    model,
                    prompt,
                    temperature,
                    max_tokens,
                    obs.max_score,
                    llm_enabled=bool(getattr(llm, "_aegis_llm_enabled", True)),
                )
            except Exception as e:
                err = f"parse_error: {e!s}"
                steps_taken = step
                rewards.append(0.0)
                log_step(step, repr(text[:300]), 0.0, True, _error_column(err, None))
                history.append(f"step={step} parse_error={_one_line(str(e))}")
                last_action = None
                last_reward = 0.0
                continue

            out = env.step(action)
            r = float(getattr(out, "reward", None) or 0.0)
            rewards.append(r)
            steps_taken = step
            last_action = _action_log_str(action)
            last_reward = r
            history.append(f"step={step} action={last_action} reward={r:.2f}")
            log_step(
                step,
                repr(_action_log_str(action)),
                r,
                bool(getattr(out, "done", False)),
                _error_column(None, None),
            )
            obs = out
            # Do not break on done; we may reset and continue for additional attempts.
            if _episode_score(rewards) >= success_score_threshold:
                break

        return True, steps_taken, rewards
    except Exception as e:
        err = f"{type(e).__name__}: {e!s}"
        steps_taken = max(steps_taken, 1)
        rewards.append(0.0)
        log_step(steps_taken, "<none>", 0.0, True, _error_column(err, None))
        return False, steps_taken, rewards
    finally:
        env.close()


async def run_remote_episode_async(
    task_name: str,
    benchmark: str,
    model: str,
    llm: OpenAI,
    docker_image: str,
    openenv_base_url: str,
    max_steps: int,
    temperature: float,
    max_tokens: int,
    episode_seed: Optional[int],
    success_score_threshold: float,
) -> Tuple[bool, int, List[float]]:
    """WebSocket client: Docker image or HTTP base URL."""
    from client import AegisEnv

    rewards: List[float] = []
    steps_taken = 0
    history: List[str] = []
    last_action: Optional[str] = None
    last_reward = 0.0
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

        for step in range(1, max_steps + 1):
            # AEGIS-Env can be a single-step episode (done after one step).
            # We keep running new episodes (reset -> step) until success threshold
            # is met or max_steps is exhausted.
            result = await env.reset(seed=episode_seed, task_name=task_name)

            obs = result.observation
            prompt = build_user_prompt(step, last_action, last_reward, history, obs)
            text = ""
            try:
                action, text = _get_action_with_retry(
                    llm,
                    model,
                    prompt,
                    temperature,
                    max_tokens,
                    obs.max_score,
                    llm_enabled=bool(getattr(llm, "_aegis_llm_enabled", True)),
                )
            except Exception as e:
                err = f"parse_error: {e!s}"
                steps_taken = step
                rewards.append(0.0)
                log_step(step, repr(text[:300]), 0.0, True, _error_column(err, None))
                history.append(f"step={step} parse_error={_one_line(str(e))}")
                last_action = None
                last_reward = 0.0
                continue

            result = await env.step(action)
            r = float(result.reward or 0.0)
            rewards.append(r)
            steps_taken = step
            last_action = _action_log_str(action)
            last_reward = r
            history.append(f"step={step} action={last_action} reward={r:.2f}")
            log_step(
                step,
                repr(_action_log_str(action)),
                r,
                result.done,
                _error_column(None, result),
            )
            # Do not break on done; we may reset and continue for additional attempts.
            if _episode_score(rewards) >= success_score_threshold:
                break

        return True, steps_taken, rewards
    except Exception as e:
        # If we already met target score, treat late transport failures as non-fatal.
        if rewards and _episode_score(rewards) >= success_score_threshold:
            return True, max(steps_taken, 1), rewards
        err = f"{type(e).__name__}: {e!s}"
        steps_taken = max(steps_taken, 1)
        if not rewards:
            rewards.append(0.0)
        log_step(steps_taken, "<none>", 0.0, True, _error_column(err, None))
        return False, steps_taken, rewards
    finally:
        if env is not None:
            with contextlib.suppress(asyncio.CancelledError, Exception):
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
            args.max_steps,
            args.temperature,
            args.max_tokens,
            _episode_seed(args),
            float(args.success_score_threshold),
        )
    return await run_remote_episode_async(
        args.task_name,
        args.benchmark,
        model,
        llm,
        docker_image,
        args.openenv_base_url,
        args.max_steps,
        args.temperature,
        args.max_tokens,
        _episode_seed(args),
        float(args.success_score_threshold),
    )

async def main_async(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="AEGIS-Env inference (OpenAI client + async OpenEnv)",
    )
    p.add_argument(
        "--task-name",
        default=TASK_NAME,
        help="Logged task name",
    )
    p.add_argument(
        "--benchmark",
        default=BENCHMARK,
        help="Logged benchmark name",
    )
    p.add_argument("--model", default=None, help="Override MODEL_NAME")
    p.add_argument("--api-base-url", default=None, dest="api_base_url")
    p.add_argument("--openenv-base-url", default=OPENENV_BASE_URL)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--success-score-threshold", type=float, default=SUCCESS_SCORE_THRESHOLD)
    p.add_argument("--episode-seed", type=int, default=None)
    p.add_argument(
        "--single-task",
        action="store_true",
        help="Run only the selected --task-name (disable default easy/medium/hard sweep).",
    )
    p.add_argument(
        "--docker-image",
        default=None,
        help="Override LOCAL_IMAGE_NAME for from_docker_image()",
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
        api_key = _get_api_key()
        api_base = _llm_api_base_url(args.api_base_url)
        model = _model_name(args.model)
        llm = OpenAI(base_url=api_base, api_key=api_key or "no-key")
        llm._aegis_llm_enabled = bool(api_key)  # type: ignore[attr-defined]
        # By default, run one example for each task tier to satisfy validators that
        # require multiple tasks with graders.
        tasks = [args.task_name] if args.single_task else ["easy", "medium", "hard"]

        # Each task is its own episode block with its own START/STEP/END lines.
        overall_success = True
        overall_exit_code = 0
        for t in tasks:
            per_steps = 0
            per_rewards: List[float] = []
            per_success = False
            try:
                args.task_name = t
                _, per_steps, per_rewards = await run_episode_async(args, model, llm)
                per_score = _episode_score(per_rewards)
                per_success = per_score >= float(args.success_score_threshold)
            except Exception as e:
                # Ensure we always emit END even on errors.
                print(f"fatal_error={type(e).__name__}: {e!s}", file=sys.stderr)
                per_steps = max(per_steps, 1)
                if not per_rewards:
                    per_rewards = [0.0]
                per_success = False
                per_score = _episode_score(per_rewards)
            finally:
                log_end(
                    per_success,
                    per_steps,
                    _episode_score(per_rewards),
                    per_rewards,
                )

            overall_success = overall_success and per_success
            overall_exit_code = 0 if overall_success else 1

        # Populate legacy outputs for completeness (not used for logging now).
        success = overall_success
        exit_code = overall_exit_code
    except SystemExit:
        raise
    except KeyboardInterrupt:
        success = False
        exit_code = 130
    except Exception as e:
        print(f"fatal_error={type(e).__name__}: {e!s}", file=sys.stderr)
        success = False
        exit_code = 1
    finally:
        # END lines are emitted per-task above; nothing to do here.
        pass

    return exit_code


if __name__ == "__main__":
    asyncio.run(main_async())

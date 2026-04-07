# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AEGIS-Env: automated grading simulation with deterministic rewards.

The dataset is downloaded from Hugging Face and cached on disk; ``reset`` and
``step`` are CPU-only.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import AegisAction, AegisObservation
except ImportError:
    from models import AegisAction, AegisObservation


def _rubrics_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _jaccard(a_text: str, b_text: str) -> float:
    a_tokens = set(a_text.lower().split())
    b_tokens = set(b_text.lower().split())
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    inter = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return float(inter) / float(union) if union else 0.0


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(override=True)
    except Exception:
        pass


def _cache_dir() -> Path:
    # Default to a repo-local cache so it works in sandboxed runners.
    # You can override via AEGIS_CACHE_DIR / HF_HOME / XDG_CACHE_HOME.
    root = (
        os.environ.get("AEGIS_CACHE_DIR")
        or os.environ.get("HF_HOME")
        or os.environ.get("XDG_CACHE_HOME")
    )
    if root:
        return Path(root) / "aegis_env"
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / ".cache" / "aegis_env"


def _unwrap_object_id(v: Any) -> str:
    # Expected schema: {"$oid": "..."}; tolerate already-string ids.
    if isinstance(v, dict) and "$oid" in v:
        return str(v.get("$oid") or "")
    return str(v or "")


def _unwrap_number(v: Any) -> Optional[float]:
    # Expected schema: number OR {"$numberDouble": "Infinity"/"-Infinity"/"NaN"}.
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict) and "$numberDouble" in v:
        s = str(v.get("$numberDouble"))
        if s == "Infinity":
            return float("inf")
        if s == "-Infinity":
            return float("-inf")
        if s == "NaN":
            return float("nan")
    try:
        return float(v)
    except Exception:
        return None


def _reference_feedback_from_record(rec: Dict[str, Any]) -> str:
    # New schema stores feedback under evaluation.agent_feedback.
    ev = rec.get("evaluation") or {}
    agent_feedback = (ev.get("agent_feedback") or {}) if isinstance(ev, dict) else {}
    if isinstance(agent_feedback, dict):
        sj = agent_feedback.get("score_justification")
        ia = agent_feedback.get("improvement_advice")
        joined = " ".join([str(x).strip() for x in [sj, ia] if x is not None]).strip()
        if joined:
            return joined
    # Backward-compat: old field name.
    return str(rec.get("reference_feedback") or "")


def _download_dataset_json(repo_id: str, filename: str, revision: Optional[str]) -> Path:
    from huggingface_hub import hf_hub_download  # type: ignore[import-not-found]

    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(cache_dir / "hf"),
        )
    except Exception:
        # In some sandboxed environments, network access to Hugging Face may be blocked.
        # If the file is already present in the global HF cache, fall back to it.
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=None,
            local_files_only=True,
        )
    stable_path = cache_dir / f"{repo_id.replace('/', '__')}__{filename}"
    try:
        stable_path.write_bytes(Path(downloaded).read_bytes())
        return stable_path
    except Exception:
        return Path(downloaded)


def _load_dataset_records() -> List[Dict[str, Any]]:
    _load_dotenv_if_available()

    repo_id = os.environ.get("AEGIS_HF_DATASET_REPO") or "NishithP2004/AEGIS-Eval-v2"
    filename = os.environ.get("AEGIS_HF_DATASET_FILE") or "dataset.json"
    revision = os.environ.get("AEGIS_HF_DATASET_REVISION") or None
    offline = str(os.environ.get("AEGIS_HF_OFFLINE") or "").lower() in {"1", "true", "yes"}

    cache_dir = _cache_dir()
    stable_path = cache_dir / f"{repo_id.replace('/', '__')}__{filename}"

    path: Optional[Path] = None
    if stable_path.exists():
        path = stable_path
    elif not offline:
        path = _download_dataset_json(repo_id, filename, revision)

    if path is None or not path.exists():
        raise RuntimeError(
            f"Dataset cache not found. Expected {stable_path}. "
            f"Set AEGIS_HF_OFFLINE=0 to allow download or provide the cached file."
        )

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        records = data["data"]
    elif isinstance(data, list):
        records = data
    else:
        raise RuntimeError(f"Unexpected dataset.json shape in {path}")

    out: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        norm: Dict[str, Any] = dict(rec)
        norm["_id"] = _unwrap_object_id(rec.get("_id"))
        for k in ("max_score", "min_score", "obtained_score"):
            norm[k] = _unwrap_number(rec.get(k))
        out.append(norm)
    return out


class AegisEnvironment(Environment[AegisAction, AegisObservation, State]):
    """
    Single-step grading episode: reset samples a row; step scores the agent output.

    Reward is in [0, 1] from accuracy (max 0.7) plus feedback overlap (max 0.3).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.dataset: List[Dict[str, Any]] = []
        self._load_error: Optional[str] = None

        self._rng = random.Random()

        self._current_record: Optional[Dict[str, Any]] = None
        self.current_ground_truth: Optional[float] = None
        self.current_reference_feedback: str = ""
        self.current_max_score: float = 1.0

        try:
            self.dataset = _load_dataset_records()
        except Exception as e:
            self._load_error = f"Dataset load failed: {e!s}"
            self.dataset = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AegisObservation:
        # OpenEnv passes the task via environment variable (or via reset kwargs in some harnesses).
        # Default to 'easy'.
        task_name = str(kwargs.get("task_name") or os.getenv("TASK_NAME", "easy"))

        # Map the task ID to the corresponding dataset label
        task_mapping = {
            "easy": "mohler",
            "medium": "asap-sas",
            "hard": "ricechem",
        }

        target_dataset = task_mapping.get(task_name, "mohler")

        if not self.dataset:
            msg = self._load_error or "No grading records loaded."
            raise RuntimeError(msg)

        # Filter the in-memory dataset loaded during __init__
        available_records = [r for r in self.dataset if r.get("dataset") == target_dataset]

        # Fallback if the dataset filter yields an empty list
        if not available_records:
            available_records = self.dataset

        # Sample a random question from the targeted difficulty tier
        selected_record = random.choice(available_records)
        self._current_record = selected_record

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        if seed is not None:
            self._rng.seed(seed)

        # Store the ground truth for the deterministic reward calculation in step()
        self.current_ground_truth = float(selected_record.get("obtained_score", 0.0))
        self.current_reference_feedback = _reference_feedback_from_record(selected_record)
        self.current_max_score = float(selected_record.get("max_score", 1.0) or 1.0)

        # Ensure rubrics are handled even if missing (like in ASAP-SAS)
        rubric_text = selected_record.get("rubrics", "")
        if not rubric_text or str(rubric_text).lower() == "nan":
            rubric_text = "Holistic grading: Evaluate the answer based on general scientific/linguistic accuracy."

        # Return the Pydantic Observation model
        return AegisObservation(
            question=selected_record.get("question", "") or "",
            rubric=_rubrics_to_str(rubric_text),
            max_score=self.current_max_score,
            student_answer=selected_record.get("student_response", "") or "",
            done=False,
            reward=None,
            grading_info={},
        )

    def step(
        self,
        action: AegisAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AegisObservation:
        if self._current_record is None:
            raise RuntimeError("Environment must be reset before step().")

        self._state.step_count += 1

        max_score = float(self._current_record.get("max_score") or 0.0)
        obtained = float(self.current_ground_truth if self.current_ground_truth is not None else 0.0)

        accuracy_reward = 0.0
        if max_score > 0.0:
            norm_human = obtained / max_score
            norm_agent = action.final_score / max_score
            if action.final_score < 0.0 or action.final_score > max_score:
                accuracy_reward = 0.0
            else:
                accuracy_reward = 0.7 * (1.0 - abs(norm_agent - norm_human))
        else:
            accuracy_reward = 0.0

        agent_text = f"{action.score_justification} {action.improvement_advice}".strip()
        word_count = len(agent_text.split())
        feedback_reward = 0.0
        if word_count >= 10:
            feedback_reward = 0.3 * _jaccard(agent_text, self.current_reference_feedback)

        total = accuracy_reward + feedback_reward
        total = max(0.0, min(1.0, total))

        info: Dict[str, Any] = {
            "accuracy_reward": accuracy_reward,
            "feedback_reward": feedback_reward,
            "total_reward": total,
            "word_count": word_count,
        }

        return AegisObservation(
            question=str(self._current_record.get("question", "")),
            rubric=_rubrics_to_str(self._current_record.get("rubrics")),
            max_score=max_score,
            student_answer=str(self._current_record.get("student_response", "")),
            done=True,
            reward=total,
            grading_info=info,
        )

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="AEGIS-Env",
            description=(
                "Automated Evaluation & Grading Intelligent System: grade student answers "
                "from question, rubric, and response with deterministic rewards."
            ),
            version="0.1.0",
        )

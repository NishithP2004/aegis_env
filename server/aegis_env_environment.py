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


def _coerce_aegis_action(action: Any) -> AegisAction:
    """
    Build this module's AegisAction.

    Callers (e.g. inference.py) may construct AegisAction from another import path;
    Pydantic v2 then treats that object as a different type than our AegisAction.
    """
    if isinstance(action, AegisAction):
        return action
    if isinstance(action, dict):
        return AegisAction.model_validate(action)
    if hasattr(action, "model_dump"):
        return AegisAction.model_validate(action.model_dump())
    return AegisAction.model_validate(action)


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
        ia = agent_feedback.get("improvement_advice")
        if ia is not None:
            return str(ia).strip()
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


def _apply_train_test_split(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply train/test split with balanced sampling across unique questions per dataset.
    Test split ratios: ricechem=10%, asap-sas=20%, mohler=30%.
    Returns only the training subset.
    """
    SEED = 42
    train_seed = random.Random(SEED)

    splits = {
        "ricechem": 0.10,
        "asap-sas": 0.20,
        "mohler": 0.30,
    }

    test_indices = set()

    for dataset, ratio in splits.items():
        # Filter records for this dataset
        dataset_indices = [i for i, rec in enumerate(records) if rec.get("dataset") == dataset]
        if not dataset_indices:
            continue

        # Group by question
        question_groups: Dict[str, List[int]] = {}
        for idx in dataset_indices:
            question = records[idx].get("question", "")
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(idx)

        # Calculate how many samples per question
        total_rows = len(dataset_indices)
        test_size = int(total_rows * ratio)
        num_questions = len(question_groups)
        per_question = max(1, int(test_size / num_questions))

        # Sample from each question to balance the test set
        for question, indices in question_groups.items():
            sample_size = min(per_question, len(indices))
            sampled_indices = train_seed.sample(indices, k=sample_size)
            test_indices.update(sampled_indices)

    # Return only training records (excluding test indices)
    train_records = [rec for i, rec in enumerate(records) if i not in test_indices]
    return train_records


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

    # Apply train/test split to return only training data
    train_records = _apply_train_test_split(out)
    return train_records


class AegisEnvironment(Environment[AegisAction, AegisObservation, State]):
    """
    Multi-step grading episode: reset samples a row; step advances a 4-stage pipeline.

    Stages: arbiter -> scrutinizer -> validator -> mentor -> finished
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

        # Multi-step pipeline state (initialized in reset()).
        self.max_iterations: int = 2
        self.refinement_loops_taken: int = 0
        self.current_stage: str = "arbiter"
        self.pipeline_history: str = ""
        self.flow_bank: float = 0.10

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

        # Initialize multi-step state machine
        self.max_iterations = 2
        self.refinement_loops_taken = 0
        self.current_stage = "arbiter"
        self.pipeline_history = "--- PIPELINE INITIATED ---\n"
        self.flow_bank = 0.10

        # Return the Pydantic Observation model
        return AegisObservation(
            question=selected_record.get("question", "") or "",
            rubric=_rubrics_to_str(rubric_text),
            max_score=self.current_max_score,
            student_answer=selected_record.get("student_response", "") or "",
            current_stage=self.current_stage,
            refinement_loops_taken=self.refinement_loops_taken,
            pipeline_history=self.pipeline_history,
            done=False,
            reward=0.0,
            grading_info={},
            metadata={},
        )

    def step(
        self,
        action: Any,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> AegisObservation:
        if self._current_record is None:
            raise RuntimeError("Environment must be reset before step().")

        action = _coerce_aegis_action(action)

        self._state.step_count += 1

        # 1) Log the agent's work into the pipeline history
        self.pipeline_history += (
            f"\n[{self.current_stage.upper()}]: "
            f"Score: {action.proposed_score} | Routing: {action.routing_decision}\n"
            f"Reasoning: {action.agent_reasoning}\n"
        )

        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        # 2) State machine routing + reward computation
        if self.current_stage == "arbiter":
            self.current_stage = "scrutinizer"
            reward = 0.02
            self.flow_bank -= reward

        elif self.current_stage == "scrutinizer":
            self.current_stage = "validator"
            reward = 0.02
            self.flow_bank -= reward

        elif self.current_stage == "validator":
            decision = str(action.routing_decision or "").strip().lower()
            if decision == "revise":
                self.refinement_loops_taken += 1
                if self.refinement_loops_taken > self.max_iterations:
                    done = True
                    reward = 0.0
                    self.pipeline_history += "\n[SYSTEM]: FATAL ERROR - Max Refinement Iterations Exceeded."
                else:
                    self.current_stage = "scrutinizer"
                    reward = 0.01
                    self.flow_bank -= reward
            elif decision == "proceed":
                self.current_stage = "mentor"
                reward = 0.02
                self.flow_bank -= reward
            else:
                done = True
                reward = 0.0
                self.pipeline_history += f"\n[SYSTEM]: FATAL ERROR - Invalid routing_decision '{decision}'."

        elif self.current_stage == "mentor":
            done = True
            max_score = float(self.current_max_score or 0.0)
            obtained = float(self.current_ground_truth if self.current_ground_truth is not None else 0.0)
            norm_human = obtained / max_score if max_score > 0 else 0.0
            norm_agent = action.proposed_score / max_score if max_score > 0 else 0.0

            # Accuracy reward (max 0.6)
            acc_reward = 0.0
            if max_score > 0 and 0.0 <= float(action.proposed_score) <= max_score:
                acc_reward = 0.6 * (1.0 - abs(norm_agent - norm_human))

            # Validity reward (max 0.3)
            feed_reward = 0.0
            if len(str(action.agent_reasoning or "").split()) >= 10:
                feed_reward = 0.3 * _jaccard(str(action.agent_reasoning or ""), self.current_reference_feedback)

            # Final Payout = Accuracy + Validity + Remaining Flow Bank
            reward = acc_reward + feed_reward + self.flow_bank
            reward = max(0.0, min(1.0, reward))

            info = {
                "accuracy_reward": acc_reward,
                "validity_reward": feed_reward,
                "flow_bank_payout": float(self.flow_bank),
                "total_step_reward": reward,
            }
        else:
            # Unknown stage: terminate safely.
            done = True
            reward = 0.0
            self.pipeline_history += "\n[SYSTEM]: ERROR - Unknown stage."

        # 3) Record reward history in the pipeline log (including intermediate 0.0).
        self.pipeline_history += (
            f"[SYSTEM]: reward={reward:.4f} done={str(done).lower()} next_stage="
            f"{self.current_stage if not done else 'finished'}\n"
        )

        rubric_text = _rubrics_to_str(self._current_record.get("rubrics"))
        return AegisObservation(
            question=str(self._current_record.get("question", "")),
            rubric=rubric_text,
            max_score=float(self.current_max_score or 1.0),
            student_answer=str(self._current_record.get("student_response", "")),
            current_stage=self.current_stage if not done else "finished",
            refinement_loops_taken=self.refinement_loops_taken,
            pipeline_history=self.pipeline_history,
            done=done,
            reward=reward,
            grading_info=info,
            metadata={},
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

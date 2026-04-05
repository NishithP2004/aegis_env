# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AEGIS-Env: automated grading simulation with deterministic rewards.

MongoDB is queried only in ``__init__``; ``reset`` and ``step`` are CPU-only.
"""

from __future__ import annotations

import os
import random
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

        uri = os.environ.get("MONGO_URI")
        if not uri:
            self._load_error = "MONGO_URI is not set; dataset is empty."
            return

        try:
            from pymongo import MongoClient  # type: ignore[import-untyped]

            client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
            try:
                coll = client["AEGIS"]["AEGIS-Eval-v2"]
                projection = {
                    "question": 1,
                    "rubrics": 1,
                    "student_response": 1,
                    "max_score": 1,
                    "obtained_score": 1,
                    "reference_feedback": 1,
                }
                cursor = coll.find({}, projection)
                for doc in cursor:
                    self.dataset.append(doc)
            finally:
                client.close()
        except Exception as e:
            self._load_error = f"MongoDB load failed: {e!s}"
            self.dataset = []

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AegisObservation:
        if seed is not None:
            self._rng.seed(seed)

        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        if not self.dataset:
            msg = self._load_error or "No grading records loaded."
            raise RuntimeError(msg)

        self._current_record = self._rng.choice(self.dataset)
        assert self._current_record is not None

        ms = float(self._current_record.get("max_score") or 0.0)
        if ms <= 0.0:
            raise RuntimeError("Sampled record has invalid max_score; cannot build observation.")

        self.current_ground_truth = float(self._current_record.get("obtained_score", 0.0))
        ref = self._current_record.get("reference_feedback")
        self.current_reference_feedback = str(ref) if ref is not None else ""

        return AegisObservation(
            question=str(self._current_record.get("question", "")),
            rubric=_rubrics_to_str(self._current_record.get("rubrics")),
            max_score=ms,
            student_answer=str(self._current_record.get("student_response", "")),
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

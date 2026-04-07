# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AEGIS-Env Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import AegisAction, AegisObservation
except ImportError:
    from models import AegisAction, AegisObservation


class AegisEnv(
    EnvClient[AegisAction, AegisObservation, State]
):
    """
    WebSocket client for the AEGIS-Env grading environment.

    Example:
        >>> with AegisEnv(base_url="http://localhost:8000").sync() as client:
        ...     r0 = client.reset()
        ...     r1 = client.step(
        ...         AegisAction(
        ...             final_score=8.0,
        ...             score_justification="...",
        ...             improvement_advice="...",
        ...         )
        ...     )
    """

    def _step_payload(self, action: AegisAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AegisObservation]:
        obs_data = payload.get("observation", {})
        metadata: Dict[str, Any] = dict(obs_data.get("metadata") or {})
        for key in ("last_action_error", "error"):
            if key in payload and payload[key] is not None:
                metadata[key] = payload[key]

        observation = AegisObservation(
            question=obs_data.get("question", ""),
            rubric=obs_data.get("rubric", ""),
            max_score=float(obs_data.get("max_score", 1.0)),
            student_answer=obs_data.get("student_answer", ""),
            grading_info=obs_data.get("grading_info") or {},
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=metadata,
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

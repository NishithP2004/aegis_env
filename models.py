# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models for AEGIS-Env (Automated Evaluation & Grading Intelligent System).

Observation and Action are strictly typed for OpenEnv and hackathon validation.
"""

from typing import Any, Dict

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AegisAction(Action):
    """Agent grading output: score, justification, and improvement advice."""

    final_score: float = Field(..., description="Assigned score within [0, max_score]")
    score_justification: str = Field(..., description="Justification for the score")
    improvement_advice: str = Field(..., description="Actionable feedback for the student")


class AegisObservation(Observation):
    """
    What the agent sees: question, rubric, caps, and the student's answer.

    Ground-truth fields are never exposed here; they live only in server-side state.
    """

    question: str = Field(..., description="Assessment question or prompt")
    rubric: str = Field(..., description="Grading rubric text")
    max_score: float = Field(..., gt=0, description="Maximum achievable score for this item")
    student_answer: str = Field(..., description="Student response to evaluate")
    grading_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic grading diagnostics from the last step (empty at reset)",
    )

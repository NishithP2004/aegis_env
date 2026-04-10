# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pydantic models for AEGIS-Env (Automated Evaluation & Grading Intelligent System).

This environment is a multi-step state machine that mimics a 4-stage grading pipeline:
arbiter -> scrutinizer -> validator -> mentor -> finished.
"""

from typing import Any, Dict, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class AegisObservation(Observation):
    question: str = Field(..., description="Assessment question or prompt.")
    rubric: str = Field(..., description="Grading rubric text.")
    max_score: float = Field(..., gt=0, description="Maximum achievable score for this item.")
    student_answer: str = Field(..., description="Student response to evaluate.")

    # Backward-compatible defaults: older servers/clients may omit these fields.
    current_stage: str = Field(
        default="arbiter",
        description="Current pipeline stage: arbiter, scrutinizer, validator, mentor, or finished.",
    )
    refinement_loops_taken: int = Field(
        default=0,
        ge=0,
        description="Number of validator-requested refinement loops taken so far.",
    )
    pipeline_history: str = Field(
        default="",
        description="Accumulated pipeline transcript across stages (includes stage outputs and reward history).",
    )

    done: bool = Field(default=False, description="Whether the episode is complete.")
    reward: Optional[float] = Field(
        default=None,
        description="Reward for this transition (typically 0.0 for intermediate stages; final reward on completion).",
    )
    grading_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Deterministic grading diagnostics from the last step (empty at reset).",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the environment (optional).",
    )


class AegisAction(Action):
    proposed_score: float = Field(
        ...,
        description="Proposed score for the student answer (in [0, max_score]).",
    )
    agent_reasoning: str = Field(
        ...,
        description="Stage-specific reasoning, critique, or feedback text.",
    )
    routing_decision: str = Field(
        default="proceed",
        description="Must be 'proceed' or 'revise'. Only matters during the 'validator' stage.",
    )

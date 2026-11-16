"""
Stage 2 — Contextual Question Engine.

Holds the answers to the *consequential* questions (the ones that actually
change the recommended pipeline). This is stored as explicit project
configuration — never used to silently change data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    ANALYTICS = "analytics"  # no downstream model — exploratory only


class Priority(str, Enum):
    PERFORMANCE = "performance"
    INTERPRETABILITY = "interpretability"
    BALANCED = "balanced"


@dataclass
class UserPreferences:
    target_column: str | None
    task_type: TaskType
    priority: Priority = Priority.BALANCED
    preserve_meaningful_missingness: bool = True

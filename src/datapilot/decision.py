"""
Decision-record schema — the core "no silent AI decisions" rule from the
proposal. Every non-trivial transformation the planner proposes must be
wrapped in one of these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionRecord:
    what: str            # e.g. "Drop an identifier-like feature"
    why: str              # e.g. "98.7% unique values, no repeated groups"
    evidence: str         # e.g. "High cardinality + identifier pattern"
    alternatives: str     # e.g. "Keep for record tracking, exclude from matrix"
    risk: str              # e.g. "Could contain hidden domain signal"
    effect: str            # e.g. "81 -> 80 candidate features"
    reversible: bool

    def as_markdown_row(self) -> str:
        return (
            f"| {self.what} | {self.why} | {self.evidence} | {self.alternatives} "
            f"| {self.risk} | {self.effect} | {'Yes' if self.reversible else 'No'} |"
        )

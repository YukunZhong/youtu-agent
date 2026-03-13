"""
Strict exact-match verifier for VQA evaluation.

Used for testing / continual evaluation only.
Returns reward=1.0 if pred == gt exactly, else 0.0.
"""

from __future__ import annotations

from utu.db import EvaluationSample

def _extract_answer(response: str) -> str:
    """Extract the answer from model response.

    MoDE format: model outputs the answer directly
    (e.g. a single letter for MCQ, or a short phrase for open VQA).
    Takes the last non-empty line, stripped of whitespace.
    """
    if not response:
        return ""
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


def verify_func(sample: EvaluationSample, timeout_score: float = 0, **kwargs) -> dict:
    """Strict exact-match verifier.

    Args:
        sample: EvaluationSample with response and correct_answer.
        timeout_score: Unused.

    Returns:
        dict with 'reward' (0.0 or 1.0) and 'reasoning'.
    """
    pred_raw = _extract_answer(sample.response or "")
    gt_raw = sample.correct_answer or ""

    if pred_raw == gt_raw:
        reward = 1.0
        reasoning = f"EXACT match: pred={pred_raw!r} == gt={gt_raw!r}"
    else:
        reward = 0.0
        reasoning = f"NO match: pred={pred_raw!r} != gt={gt_raw!r}"

    return {"reward": reward, "reasoning": reasoning}

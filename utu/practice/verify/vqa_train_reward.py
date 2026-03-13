"""
Hierarchical continuous reward verifier for VQA training.

Tiers (descending):
  EXACT  -> 1.00
  CANON  -> 0.93 + 0.02 * char_sim
  ALIAS  -> 0.80 + 0.04 * char_sim
  SYN    -> 0.68 + 0.04 * char_sim
  SEM    -> 0.40 + 0.20 * clip((sem - 0.85)/0.15, 0, 1)
  MISS   -> 0.00

Answer-type gating:
  - ScienceQA (mcq / option_letter): EXACT / CANON / ALIAS only
  - ImageNet (open_label): EXACT / CANON / ALIAS / controlled SYN
  - GQA:
      open_label / open_phrase -> all tiers
      count / boolean / color / direction -> no SEM
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from utu.db import EvaluationSample

# ──────────────────────────── string helpers ────────────────────────────

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_MULTISPACE = re.compile(r"\s+")


def _canon(text: str) -> str:
    """Canonicalize: lower, strip accents, remove articles & punctuation, collapse spaces."""
    text = text.strip().lower()
    # strip accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # remove articles
    text = _ARTICLES.sub(" ", text)
    # remove punctuation
    text = _PUNCT.sub(" ", text)
    text = _MULTISPACE.sub(" ", text).strip()
    return text


def _char_sim(a: str, b: str) -> float:
    """Character-level similarity via SequenceMatcher."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _sem_sim(a: str, b: str) -> float:
    """Simple semantic similarity fallback (word overlap Jaccard).

    A lightweight proxy when no embedding model is available.
    Replace with CLIP/sentence-transformer cosine if needed.
    """
    sa = set(_canon(a).split())
    sb = set(_canon(b).split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ──────────────────────────── alias / synonym / semantic ────────────────────────────

# Controlled synonym pairs for ImageNet-style labels
_IMAGENET_SYNONYMS: dict[str, set[str]] = {
    "dog": {"canine", "hound", "pup", "puppy"},
    "cat": {"feline", "kitten", "kitty"},
    "car": {"automobile", "vehicle"},
    "airplane": {"plane", "aeroplane", "aircraft"},
    "bird": {"avian"},
    "boat": {"ship", "vessel"},
    "television": {"tv", "telly"},
    "cellphone": {"cell phone", "mobile phone", "smartphone"},
    "laptop": {"notebook computer"},
}

# Build reverse map
_SYN_REVERSE: dict[str, str] = {}
for _key, _vals in _IMAGENET_SYNONYMS.items():
    for _v in _vals:
        _SYN_REVERSE[_v] = _key
    _SYN_REVERSE[_key] = _key


def _alias_match(pred_canon: str, gt_canon: str, aliases: list[str] | None) -> bool:
    """Check if pred matches any alias (after canonicalization)."""
    if not aliases:
        return False
    for alias in aliases:
        if _canon(alias) == pred_canon:
            return True
    return False


def _synonym_match(pred_canon: str, gt_canon: str, task: str) -> bool:
    """Check controlled synonym match."""
    p_root = _SYN_REVERSE.get(pred_canon)
    g_root = _SYN_REVERSE.get(gt_canon)
    if p_root and g_root and p_root == g_root and pred_canon != gt_canon:
        return True
    return False


def _semantic_match(pred_canon: str, gt_canon: str, threshold: float = 0.85) -> tuple[bool, float]:
    """Check semantic similarity above threshold."""
    sim = _sem_sim(pred_canon, gt_canon)
    return sim >= threshold, sim


# ──────────────────────────── gating logic ────────────────────────────

def _get_enabled_tiers(task: str, answer_type: str) -> set[str]:
    """Return set of enabled tier names based on task and answer_type."""
    # Always enable these
    enabled = {"EXACT", "CANON", "ALIAS"}

    if task == "scienceqa":
        # MCQ: only exact / canon / alias
        pass
    elif task == "imagenet":
        # Controlled SYN only
        enabled.add("SYN")
    elif task == "gqa":
        if answer_type in ("open_label", "open_phrase"):
            enabled.add("SYN")
            enabled.add("SEM")
        elif answer_type in ("count", "boolean", "color", "direction"):
            enabled.add("SYN")
            # SEM disabled for these structured types
        else:
            enabled.add("SYN")
    else:
        # Default: enable all
        enabled.update({"SYN", "SEM"})

    return enabled


# ──────────────────────────── main verifier ────────────────────────────

def _compute_reward(
    pred_raw: str,
    gt_raw: str,
    task: str = "",
    answer_type: str = "",
    aliases: list[str] | None = None,
) -> dict:
    """Compute hierarchical continuous reward.

    Returns dict with keys: reward, tier, reasoning.
    """
    enabled = _get_enabled_tiers(task, answer_type)
    pred_canon = _canon(pred_raw)
    gt_canon = _canon(gt_raw)

    # Tier 1: EXACT
    if pred_raw == gt_raw:
        return _build_result(1.0, "EXACT", pred_raw, gt_raw, pred_canon, gt_canon)

    # Tier 2: CANON
    if "CANON" in enabled and pred_canon == gt_canon:
        sim = _char_sim(pred_raw, gt_raw)
        reward = 0.93 + 0.02 * sim
        return _build_result(reward, "CANON", pred_raw, gt_raw, pred_canon, gt_canon)

    # Tier 3: ALIAS
    if "ALIAS" in enabled and _alias_match(pred_canon, gt_canon, aliases):
        sim = _char_sim(pred_canon, gt_canon)
        reward = 0.80 + 0.04 * sim
        return _build_result(reward, "ALIAS", pred_raw, gt_raw, pred_canon, gt_canon)

    # Tier 4: SYN
    if "SYN" in enabled and _synonym_match(pred_canon, gt_canon, task):
        sim = _char_sim(pred_canon, gt_canon)
        reward = 0.68 + 0.04 * sim
        return _build_result(reward, "SYN", pred_raw, gt_raw, pred_canon, gt_canon)

    # Tier 5: SEM
    if "SEM" in enabled:
        is_sem, sem_score = _semantic_match(pred_canon, gt_canon)
        if is_sem:
            clamped = max(0.0, min(1.0, (sem_score - 0.85) / 0.15))
            reward = 0.40 + 0.20 * clamped
            return _build_result(reward, "SEM", pred_raw, gt_raw, pred_canon, gt_canon, sem_score=sem_score)

    # Tier 6: MISS
    return _build_result(0.0, "MISS", pred_raw, gt_raw, pred_canon, gt_canon)


def _build_result(
    reward: float,
    tier: str,
    pred_raw: str,
    gt_raw: str,
    pred_canon: str,
    gt_canon: str,
    sem_score: float | None = None,
) -> dict:
    reasoning_parts = [
        f"pred_raw={pred_raw!r}",
        f"gt_raw={gt_raw!r}",
        f"pred_canon={pred_canon!r}",
        f"gt_canon={gt_canon!r}",
        f"tier={tier}",
        f"reward={reward:.4f}",
    ]
    if sem_score is not None:
        reasoning_parts.append(f"sem_score={sem_score:.4f}")
    reasoning = " | ".join(reasoning_parts)
    return {"reward": reward, "tier": tier, "reasoning": reasoning}


# ──────────────────────────── extract answer ────────────────────────────

def _extract_answer(response: str) -> str:
    """Extract the answer from the model response.

    MoDE format: model outputs the answer directly
    (e.g. a single letter for MCQ, or a short phrase for open VQA).
    Takes the last non-empty line, stripped of whitespace.
    """
    if not response:
        return ""
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    return lines[-1] if lines else ""


# ──────────────────────────── public interface ────────────────────────────

def verify_func(sample: EvaluationSample, timeout_score: float = 0, **kwargs) -> dict:
    """Verify function matching youtu-agent practice verifier interface.

    Args:
        sample: EvaluationSample with response, correct_answer, meta fields.
        timeout_score: Score to assign on timeout (unused here).

    Returns:
        dict with 'reward' (float) and 'reasoning' (str).
    """
    pred_raw = _extract_answer(sample.response or "")
    gt_raw = sample.correct_answer or ""

    meta = sample.meta or {}
    task = meta.get("task", "")
    answer_type = meta.get("answer_type", "")
    aliases = meta.get("aliases", None)

    result = _compute_reward(
        pred_raw=pred_raw,
        gt_raw=gt_raw,
        task=task,
        answer_type=answer_type,
        aliases=aliases,
    )
    return {"reward": result["reward"], "reasoning": result["reasoning"]}

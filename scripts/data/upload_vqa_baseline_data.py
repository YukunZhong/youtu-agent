#!/usr/bin/env python3
"""
Upload VQA baseline data (ScienceQA / ImageNet / GQA) into the youtu-agent DB.

Reads the MoDE-official JSONL format and converts to DatasetSample
with the schema required by Training-free GRPO practice.

Usage:
    python scripts/data/upload_vqa_baseline_data.py \
        --data_base_dir /path/to/MoDE-official \
        --image_base_dir /path/to/coin_raw \
        --tasks scienceqa imagenet gqa
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from utu.db import DatasetSample, DBService
from utu.utils import SQLModelUtils

# ──────────────────────────────────────────────────────────────────
# Task configurations
# ──────────────────────────────────────────────────────────────────

TASK_CONFIG = {
    "scienceqa": {
        "train_file": "data/ScienceQA/train_data.jsonl",
        "test_file": "data/ScienceQA/test_data.jsonl",
        "train_dataset": "ScienceQA_train",
        "test_dataset": "ScienceQA_eval",
        "default_question_type": "mcq",
        "default_answer_type": "option_letter",
    },
    "imagenet": {
        "train_file": "data/ImageNet/train_data.jsonl",
        "test_file": "data/ImageNet/test_data.jsonl",
        "train_dataset": "ImageNet_train",
        "test_dataset": "ImageNet_eval",
        "default_question_type": "object",
        "default_answer_type": "open_label",
    },
    "gqa": {
        "train_file": "data/GQA/train_data.jsonl",
        "test_file": "data/GQA/test_data.jsonl",
        "train_dataset": "GQA_train",
        "test_dataset": "GQA_eval",
        "default_question_type": "relation",
        "default_answer_type": "open_phrase",
    },
}


# ──────────────────────────────────────────────────────────────────
# GQA answer-type heuristic
# ──────────────────────────────────────────────────────────────────

_BOOLEAN_ANSWERS = {"yes", "no", "true", "false"}
_COLOR_WORDS = {
    "red", "blue", "green", "yellow", "white", "black", "brown",
    "gray", "grey", "orange", "pink", "purple", "beige", "tan",
}
_DIRECTION_WORDS = {"left", "right", "top", "bottom", "above", "below", "behind", "front"}


def _infer_gqa_answer_type(answer: str, question: str) -> str:
    """Heuristically infer GQA answer_type."""
    ans_lower = answer.strip().lower()
    q_lower = question.strip().lower()

    if ans_lower in _BOOLEAN_ANSWERS:
        return "boolean"

    # count: answer is a pure number
    if re.fullmatch(r"\d+", ans_lower):
        return "count"

    if ans_lower in _COLOR_WORDS or q_lower.startswith("what color"):
        return "color"

    if ans_lower in _DIRECTION_WORDS:
        return "direction"

    return "open_phrase"


def _infer_gqa_question_type(question: str) -> str:
    """Heuristically infer GQA question_type."""
    q = question.strip().lower()
    if q.startswith(("is ", "are ", "do ", "does ", "was ", "were ", "has ", "have ", "can ")):
        return "boolean"
    if "how many" in q:
        return "counting"
    if "what color" in q or "what is the color" in q:
        return "attribute"
    if "where" in q or "which side" in q or "left" in q or "right" in q:
        return "spatial"
    return "relation"


# ──────────────────────────────────────────────────────────────────
# Conversion
# ──────────────────────────────────────────────────────────────────

def _extract_question_text(text: str) -> str:
    """Remove <image> tag and reserved tokens to get pure question text."""
    text = text.replace("<image>", "").strip()
    # Remove reserved tokens like <reserved08706>...
    text = re.sub(r"<reserved\d+>.*", "", text).strip()
    return text


def _extract_choices(text: str) -> list[str] | None:
    """Extract MCQ choices from question text if present."""
    lines = text.strip().split("\n")
    choices = []
    for line in lines:
        line = line.strip()
        if re.match(r"^[A-F]\.\s", line):
            choices.append(line)
    return choices if choices else None


def convert_jsonl_record(
    record: dict,
    task: str,
    index: int,
    image_base_dir: str,
    split: str,
) -> DatasetSample:
    """Convert a MoDE JSONL record to DatasetSample."""
    cfg = TASK_CONFIG[task]
    question_text = _extract_question_text(record["text"])
    answer = record["answer"]

    # Build sample_id
    if "question_id" in record:
        sample_id = f"{task}_{split}_{record['question_id']}"
    else:
        sample_id = f"{task}_{split}_{index}"

    # Save full model tokens (VQGAN-processed) as .npy file
    # Use 'tokens' field: [BOS, BOI, image_tokens+4 (1024), text_tokens...]
    # This is the complete Anole/Chameleon model input sequence
    model_tokens = record.get("tokens", [])
    token_save_dir = os.path.join(image_base_dir, f"{task}_tokens", split)
    os.makedirs(token_save_dir, exist_ok=True)
    token_path = os.path.join(token_save_dir, f"{index:06d}.npy")

    if model_tokens and not os.path.exists(token_path):
        import numpy as np
        np.save(token_path, np.array(model_tokens, dtype=np.int32))

    # Build meta
    meta = {
        "sample_id": sample_id,
        "task": task,
        "question_type": cfg["default_question_type"],
        "answer_type": cfg["default_answer_type"],
    }

    # Task-specific meta
    if task == "scienceqa":
        choices = _extract_choices(record["text"])
        if choices:
            meta["choices"] = choices
            meta["gt_choice"] = answer
    elif task == "imagenet":
        # answer might have trailing period
        answer_clean = answer.rstrip(".")
        meta["answer_type"] = "open_label"
    elif task == "gqa":
        meta["answer_type"] = _infer_gqa_answer_type(answer, question_text)
        meta["question_type"] = _infer_gqa_question_type(question_text)

    return DatasetSample(
        dataset="",  # will be set by caller
        index=index,
        source="training_free_grpo",
        question=question_text,
        answer=answer,
        file_name=token_path,
        meta=meta,
    )


# ──────────────────────────────────────────────────────────────────
# Upload logic
# ──────────────────────────────────────────────────────────────────

def upload_task(
    data_base_dir: str,
    image_base_dir: str,
    task: str,
    max_train: int | None = None,
    max_test: int | None = None,
):
    """Upload train + test data for one task."""
    cfg = TASK_CONFIG[task]

    for split, dataset_name, max_n in [
        ("train", cfg["train_dataset"], max_train),
        ("test", cfg["test_dataset"], max_test),
    ]:
        file_key = f"{split}_file"
        filepath = os.path.join(data_base_dir, cfg[file_key])
        if not os.path.isfile(filepath):
            print(f"  [SKIP] {filepath} not found")
            continue

        samples = []
        with open(filepath, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_n is not None and i >= max_n:
                    break
                record = json.loads(line.strip())
                sample = convert_jsonl_record(
                    record, task, i, image_base_dir, split
                )
                sample.dataset = dataset_name
                samples.append(sample)

        DBService.add(samples)
        print(f"  Uploaded {len(samples)} samples to '{dataset_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Upload VQA baseline data for Training-free GRPO practice."
    )
    parser.add_argument(
        "--data_base_dir",
        type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/MoDE-official",
        help="Path to MoDE-official directory",
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/coin_raw",
        help="Base directory for saving/loading image token files",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["scienceqa", "imagenet", "gqa"],
        choices=["scienceqa", "imagenet", "gqa"],
        help="Tasks to upload",
    )
    parser.add_argument("--max_train", type=int, default=None, help="Max training samples per task")
    parser.add_argument("--max_test", type=int, default=None, help="Max test samples per task")
    args = parser.parse_args()

    if not SQLModelUtils.check_db_available():
        print("Error: Database is not available. Check UTU_DB_URL environment variable.")
        return

    for task in args.tasks:
        print(f"\n=== Uploading {task} ===")
        upload_task(
            args.data_base_dir,
            args.image_base_dir,
            task,
            max_train=args.max_train,
            max_test=args.max_test,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

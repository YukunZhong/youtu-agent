#!/usr/bin/env python3
"""
Continual Training-free GRPO Pipeline for VQA Baseline.

Runs the three-stage continual learning pipeline:
  Stage 1: ScienceQA  -> produces E1
  Stage 2: ImageNet   -> starts from E1, produces E2
  Stage 3: GQA        -> starts from E2, produces E3

After each stage:
  - Evaluates all seen tasks (strict exact match)
  - Runs CC101 T2I forgetting evaluation

This script orchestrates the existing youtu-agent practice pipeline
by calling run_training_free_GRPO and run_eval sequentially, with
the agent config from the previous stage fed into the next.

Usage:
    python scripts/run_continual_vqa_baseline.py \
        --data_base_dir /path/to/MoDE-official \
        --cc101_dir /path/to/customconcept101 \
        --output_dir ./output_dir/continual_baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utu.config import ConfigLoader, EvalConfig
from utu.eval import BaseBenchmark
from utu.practice import TrainingFreeGRPO, parse_training_free_grpo_config


# ──────────────────────────────────────────────────────────────────
# Stage definitions
# ──────────────────────────────────────────────────────────────────

STAGES = [
    {
        "name": "scienceqa",
        "practice_config": "scienceqa_practice",
        "eval_configs": ["vqa/scienceqa_eval"],
        "practice_eval_config": "vqa/scienceqa_practice_eval",
    },
    {
        "name": "imagenet",
        "practice_config": "imagenet_practice",
        "eval_configs": ["vqa/scienceqa_eval", "vqa/imagenet_eval"],
        "practice_eval_config": "vqa/imagenet_practice_eval",
    },
    {
        "name": "gqa",
        "practice_config": "gqa_practice",
        "eval_configs": ["vqa/scienceqa_eval", "vqa/imagenet_eval", "vqa/gqa_eval"],
        "practice_eval_config": "vqa/gqa_practice_eval",
    },
]


# ──────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────

async def run_eval(config_name: str, agent_config_override: str | None = None) -> dict:
    """Run a single evaluation benchmark.

    Returns dict with dataset name and accuracy.
    """
    config = ConfigLoader.load_eval_config(config_name)
    if agent_config_override:
        # Override agent config to use the practice-enhanced agent
        import yaml
        with open(agent_config_override) as f:
            agent_data = yaml.safe_load(f)
        if "agent" in agent_data:
            config.agent = ConfigLoader._build_agent_config(agent_data["agent"])

    runner = BaseBenchmark(config)
    await runner.main()

    # Collect results
    return {
        "dataset": config.data.dataset,
        "exp_id": config.exp_id,
    }


async def run_practice(config_name: str, agent_config_override: str | None = None) -> str:
    """Run Training-free GRPO practice.

    Returns path to the output agent config with experiences.
    """
    # Build config from Hydra config system
    config = ConfigLoader.load_practice_config(config_name)
    if agent_config_override:
        import yaml
        with open(agent_config_override) as f:
            agent_data = yaml.safe_load(f)
        if "agent" in agent_data:
            config.evaluation.agent = ConfigLoader._build_agent_config(agent_data["agent"])

    grpo = TrainingFreeGRPO(config)
    result = await grpo.run()
    return result


# ──────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────

async def run_pipeline(args):
    """Run the full continual learning pipeline."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    continual_results = {
        "stages": [],
        "accuracy_matrix": [],
    }

    current_agent_config = None  # None = use default base config

    for stage_idx, stage in enumerate(STAGES):
        stage_name = stage["name"]
        print(f"\n{'='*60}")
        print(f"  STAGE {stage_idx + 1}: {stage_name.upper()}")
        print(f"{'='*60}\n")

        stage_dir = os.path.join(output_dir, f"stage{stage_idx + 1}_{stage_name}")
        os.makedirs(stage_dir, exist_ok=True)

        # ─── 1) Practice ───
        print(f"\n--- Practice: {stage_name} ---")
        new_agent_config = await run_practice(
            stage["practice_config"],
            agent_config_override=current_agent_config,
        )
        print(f"  Practice complete. New agent config: {new_agent_config}")

        # Save a copy of the agent config
        if new_agent_config and os.path.isfile(new_agent_config):
            saved_config = os.path.join(stage_dir, f"{stage_name}_agent_config.yaml")
            shutil.copy2(new_agent_config, saved_config)
            current_agent_config = saved_config

        # ─── 2) Evaluate seen tasks ───
        print(f"\n--- Evaluation: seen tasks after {stage_name} ---")
        stage_eval_results = {}
        for eval_config_name in stage["eval_configs"]:
            try:
                result = await run_eval(eval_config_name, agent_config_override=current_agent_config)
                stage_eval_results[result["dataset"]] = result
                print(f"  {result['dataset']}: done")
            except Exception as e:
                print(f"  {eval_config_name}: FAILED ({e})")
                stage_eval_results[eval_config_name] = {"error": str(e)}

        # ─── 3) CC101 evaluation ───
        print(f"\n--- CC101 T2I evaluation after {stage_name} ---")
        try:
            from scripts.eval_cc101_anole import run_cc101_evaluation
            cc101_result = run_cc101_evaluation(
                cc101_dir=args.cc101_dir,
                agent_config_path=current_agent_config,
                output_dir=os.path.join(output_dir, "cc101_eval"),
                stage_name=f"stage{stage_idx + 1}_{stage_name}",
                device=args.device,
                max_concepts=args.max_cc101_concepts,
                skip_generation=args.skip_cc101_gen,
            )
        except Exception as e:
            print(f"  CC101 evaluation failed: {e}")
            cc101_result = {"error": str(e)}

        # ─── 4) Record results ───
        stage_record = {
            "stage": stage_idx + 1,
            "task": stage_name,
            "agent_config": current_agent_config,
            "vqa_eval": stage_eval_results,
            "cc101_eval": cc101_result if isinstance(cc101_result, dict) else {},
        }
        continual_results["stages"].append(stage_record)

        # Save intermediate results
        results_path = os.path.join(output_dir, "continual_results.json")
        with open(results_path, "w") as f:
            json.dump(continual_results, f, indent=2)
        print(f"\n  Stage {stage_idx + 1} results saved to {results_path}")

    # ─── Final summary ───
    print(f"\n{'='*60}")
    print("  CONTINUAL LEARNING PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\nFull results: {os.path.join(output_dir, 'continual_results.json')}")

    return continual_results


def main():
    parser = argparse.ArgumentParser(
        description="Run continual Training-free GRPO VQA baseline pipeline"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir/continual_baseline",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--cc101_dir",
        type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/customconcept101",
        help="Path to CustomConcept101 dataset",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for CLIP evaluation")
    parser.add_argument("--max_cc101_concepts", type=int, default=None, help="Max CC101 concepts")
    parser.add_argument("--skip_cc101_gen", action="store_true", help="Skip CC101 image generation")
    args = parser.parse_args()

    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()

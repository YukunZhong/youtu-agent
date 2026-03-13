#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Continual Training-free GRPO VQA Baseline Pipeline
# ──────────────────────────────────────────────────────────────
#
# This script runs the three-stage continual learning pipeline:
#   Stage 1: ScienceQA  -> produces E1
#   Stage 2: ImageNet   -> starts from E1, produces E2
#   Stage 3: GQA        -> starts from E2, produces E3
#
# After each stage, evaluates all seen tasks + CC101.
#
# Usage:
#   bash scripts/run_continual_vqa_pipeline.sh
#
# Prerequisites:
#   1. youtu-agent installed (uv sync --group dev)
#   2. .env configured with model endpoint
#   3. Data uploaded (python scripts/data/upload_vqa_baseline_data.py)
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Bypass proxy for localhost API calls
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

OUTPUT_DIR="${OUTPUT_DIR:-./output_dir/continual_baseline}"
CC101_DIR="${CC101_DIR:-/data1/data/kangborui/zhongyukun/medmax/customconcept101}"

mkdir -p "$OUTPUT_DIR"

# Use venv python
PYTHON=".venv/bin/python"
# Agent config output directory (where the installed package saves configs)
AGENT_CFG_DIR="/home/kangborui/CLProject_Suzhou/zhongyukun/youtu-agent/configs/agents/practice"

echo "============================================================"
echo "  Continual Training-free GRPO VQA Baseline"
echo "  Output: $OUTPUT_DIR"
echo "============================================================"


# ──────────────────────── Stage 0: Baseline eval ────────────────────────
echo ""
echo "--- Stage 0: Baseline evaluation (no experiences) ---"

$PYTHON scripts/run_eval.py --config_name vqa/scienceqa_eval --exp_id baseline_scienceqa 2>&1 | tee "$OUTPUT_DIR/baseline_scienceqa_eval.log"
$PYTHON scripts/run_eval.py --config_name vqa/imagenet_eval --exp_id baseline_imagenet 2>&1 | tee "$OUTPUT_DIR/baseline_imagenet_eval.log"
$PYTHON scripts/run_eval.py --config_name vqa/gqa_eval --exp_id baseline_gqa 2>&1 | tee "$OUTPUT_DIR/baseline_gqa_eval.log"

$PYTHON scripts/eval_cc101_anole.py \
    --stage_name stage0_baseline \
    --cc101_dir "$CC101_DIR" \
    --output_dir "$OUTPUT_DIR/cc101_eval" \
    2>&1 | tee "$OUTPUT_DIR/baseline_cc101_eval.log"


# ──────────────────────── Stage 1: ScienceQA ────────────────────────
echo ""
echo "============================================================"
echo "  STAGE 1: ScienceQA Practice"
echo "============================================================"

$PYTHON scripts/run_training_free_GRPO.py \
    --config_name scienceqa_practice \
    2>&1 | tee "$OUTPUT_DIR/stage1_practice.log"

STAGE1_AGENT="$AGENT_CFG_DIR/scienceqa_practice_agent.yaml"

echo ""
echo "--- Stage 1: Evaluate ScienceQA ---"
$PYTHON scripts/run_eval.py \
    --config_name vqa/scienceqa_practice_eval \
    --exp_id stage1_scienceqa \
    2>&1 | tee "$OUTPUT_DIR/stage1_scienceqa_eval.log"

$PYTHON scripts/eval_cc101_anole.py \
    --agent_config "$STAGE1_AGENT" \
    --stage_name stage1_scienceqa \
    --cc101_dir "$CC101_DIR" \
    --output_dir "$OUTPUT_DIR/cc101_eval" \
    2>&1 | tee "$OUTPUT_DIR/stage1_cc101_eval.log"


# ──────────────────────── Stage 2: ImageNet ────────────────────────
echo ""
echo "============================================================"
echo "  STAGE 2: ImageNet Practice (starting from E1)"
echo "============================================================"

$PYTHON scripts/run_training_free_GRPO.py \
    --config_name imagenet_practice \
    --agent_config "$STAGE1_AGENT" \
    2>&1 | tee "$OUTPUT_DIR/stage2_practice.log"

STAGE2_AGENT="$AGENT_CFG_DIR/imagenet_practice_agent.yaml"

echo ""
echo "--- Stage 2: Evaluate ScienceQA + ImageNet ---"
$PYTHON scripts/run_eval.py \
    --config_name vqa/scienceqa_eval \
    --exp_id stage2_scienceqa \
    2>&1 | tee "$OUTPUT_DIR/stage2_scienceqa_eval.log"

$PYTHON scripts/run_eval.py \
    --config_name vqa/imagenet_practice_eval \
    --exp_id stage2_imagenet \
    2>&1 | tee "$OUTPUT_DIR/stage2_imagenet_eval.log"

$PYTHON scripts/eval_cc101_anole.py \
    --agent_config "$STAGE2_AGENT" \
    --stage_name stage2_imagenet \
    --cc101_dir "$CC101_DIR" \
    --output_dir "$OUTPUT_DIR/cc101_eval" \
    2>&1 | tee "$OUTPUT_DIR/stage2_cc101_eval.log"


# ──────────────────────── Stage 3: GQA ────────────────────────
echo ""
echo "============================================================"
echo "  STAGE 3: GQA Practice (starting from E2)"
echo "============================================================"

$PYTHON scripts/run_training_free_GRPO.py \
    --config_name gqa_practice \
    --agent_config "$STAGE2_AGENT" \
    2>&1 | tee "$OUTPUT_DIR/stage3_practice.log"

STAGE3_AGENT="$AGENT_CFG_DIR/gqa_practice_agent.yaml"

echo ""
echo "--- Stage 3: Evaluate ScienceQA + ImageNet + GQA ---"
$PYTHON scripts/run_eval.py \
    --config_name vqa/scienceqa_eval \
    --exp_id stage3_scienceqa \
    2>&1 | tee "$OUTPUT_DIR/stage3_scienceqa_eval.log"

$PYTHON scripts/run_eval.py \
    --config_name vqa/imagenet_eval \
    --exp_id stage3_imagenet \
    2>&1 | tee "$OUTPUT_DIR/stage3_imagenet_eval.log"

$PYTHON scripts/run_eval.py \
    --config_name vqa/gqa_practice_eval \
    --exp_id stage3_gqa \
    2>&1 | tee "$OUTPUT_DIR/stage3_gqa_eval.log"

$PYTHON scripts/eval_cc101_anole.py \
    --agent_config "$STAGE3_AGENT" \
    --stage_name stage3_gqa \
    --cc101_dir "$CC101_DIR" \
    --output_dir "$OUTPUT_DIR/cc101_eval" \
    2>&1 | tee "$OUTPUT_DIR/stage3_cc101_eval.log"


# ──────────────────────── Done ────────────────────────
echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Results in: $OUTPUT_DIR"
echo "============================================================"

#!/usr/bin/env python3
"""
CustomConcept101 Text-to-Image Forgetting Evaluation.

For each practice stage, measures how much the injected VQA experiences
interfere with Anole's T2I generation capability.

Reports:
  - CLIP Text-Image similarity
  - CLIP Image-Image similarity

Usage:
    python scripts/eval_cc101_anole.py \
        --agent_config configs/agents/practice/anole_vqa_base.yaml \
        --cc101_dir /path/to/customconcept101 \
        --output_dir ./output_dir/cc101_eval \
        --stage_name stage1_scienceqa
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def load_clip_model(device: str = "cuda"):
    """Load CLIP model for evaluation."""
    try:
        import clip
        model, preprocess = clip.load("ViT-B/32", device=device)
        return model, preprocess, "openai_clip"
    except ImportError:
        pass

    try:
        from transformers import CLIPModel, CLIPProcessor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor, "hf_clip"
    except ImportError:
        raise ImportError("Please install either 'clip' (openai) or 'transformers' for CLIP evaluation.")


def clip_text_image_score(
    model,
    preprocess,
    clip_type: str,
    text: str,
    image: Image.Image,
    device: str = "cuda",
) -> float:
    """Compute CLIP text-image cosine similarity."""
    with torch.no_grad():
        if clip_type == "openai_clip":
            import clip
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_input = clip.tokenize([text], truncate=True).to(device)
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            return (image_features @ text_features.T).item()
        else:
            inputs = preprocess(text=[text], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            return (outputs.logits_per_image / 100.0).item()


def clip_image_image_score(
    model,
    preprocess,
    clip_type: str,
    image1: Image.Image,
    image2: Image.Image,
    device: str = "cuda",
) -> float:
    """Compute CLIP image-image cosine similarity."""
    with torch.no_grad():
        if clip_type == "openai_clip":
            img1_input = preprocess(image1).unsqueeze(0).to(device)
            img2_input = preprocess(image2).unsqueeze(0).to(device)
            feat1 = model.encode_image(img1_input)
            feat2 = model.encode_image(img2_input)
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            return (feat1 @ feat2.T).item()
        else:
            inputs1 = preprocess(images=image1, return_tensors="pt")
            inputs2 = preprocess(images=image2, return_tensors="pt")
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            feat1 = model.get_image_features(**inputs1)
            feat2 = model.get_image_features(**inputs2)
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            return (feat1 @ feat2.T).item()


def load_experiences_from_agent_config(agent_config_path: str) -> str:
    """Load experiences from an agent config YAML file.

    The practice module saves experiences into the agent instructions.
    We extract them to inject into the T2I prompt.
    """
    import yaml
    with open(agent_config_path, "r") as f:
        config = yaml.safe_load(f)

    instructions = ""
    if isinstance(config, dict):
        agent = config.get("agent", config)
        instructions = agent.get("instructions", "")

    # Extract experiences section
    if "Helpful experiences:" in instructions:
        idx = instructions.index("Helpful experiences:")
        experiences = instructions[idx:]
    elif "experiences" in instructions.lower():
        experiences = instructions
    else:
        experiences = ""

    return experiences


def generate_image_with_anole(
    prompt: str,
    experiences: str,
    anole_model=None,
    anole_tokenizer=None,
) -> Image.Image | None:
    """Generate an image using Anole with injected experiences.

    If Anole model is not loaded, returns None (placeholder for integration).
    """
    if anole_model is None:
        return None

    full_prompt = prompt
    if experiences:
        full_prompt = f"{experiences}\n\n{prompt}"

    # This is a placeholder - actual generation depends on Anole's API
    # The user should implement this based on their Anole deployment
    try:
        # Attempt generation via the model
        output = anole_model.generate(full_prompt)
        if hasattr(output, "image"):
            return output.image
    except Exception:
        pass

    return None


def run_cc101_evaluation(
    cc101_dir: str,
    agent_config_path: str | None,
    output_dir: str,
    stage_name: str,
    device: str = "cuda",
    max_concepts: int | None = None,
    skip_generation: bool = False,
):
    """Run CC101 T2I forgetting evaluation.

    If skip_generation=True, only compute CLIP scores from
    previously generated images in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load CC101 dataset
    dataset_path = os.path.join(cc101_dir, "dataset.json")
    with open(dataset_path) as f:
        concepts = json.load(f)

    if max_concepts is not None:
        concepts = concepts[:max_concepts]

    # Load experiences
    experiences = ""
    if agent_config_path and os.path.isfile(agent_config_path):
        experiences = load_experiences_from_agent_config(agent_config_path)
        print(f"Loaded experiences ({len(experiences)} chars) from {agent_config_path}")

    # Load CLIP
    print("Loading CLIP model...")
    clip_model, clip_preprocess, clip_type = load_clip_model(device)

    all_text_image_scores = []
    all_image_image_scores = []
    per_concept_results = []

    gen_dir = os.path.join(output_dir, stage_name, "generated")
    os.makedirs(gen_dir, exist_ok=True)

    for cidx, concept in enumerate(tqdm(concepts, desc="CC101 concepts")):
        class_prompt = concept["class_prompt"]
        instance_dir = os.path.join(cc101_dir, concept["instance_data_dir"])
        prompt_file = os.path.join(cc101_dir, concept["prompt_filename"])

        # Load reference images
        ref_images = []
        if os.path.isdir(instance_dir):
            for fname in sorted(os.listdir(instance_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    try:
                        img = Image.open(os.path.join(instance_dir, fname)).convert("RGB")
                        ref_images.append(img)
                    except Exception:
                        continue

        if not ref_images:
            continue

        # Load prompts
        prompts = []
        if os.path.isfile(prompt_file):
            with open(prompt_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Replace {} placeholder with class_prompt
                        prompts.append(line.format(class_prompt))

        if not prompts:
            prompts = [f"a photo of a {class_prompt}"]

        concept_ti_scores = []
        concept_ii_scores = []

        for pidx, prompt in enumerate(prompts[:5]):  # limit to 5 prompts per concept
            gen_path = os.path.join(gen_dir, f"concept_{cidx:03d}_prompt_{pidx:02d}.png")

            if not skip_generation:
                gen_img = generate_image_with_anole(prompt, experiences)
                if gen_img is not None:
                    gen_img.save(gen_path)
            else:
                gen_img = None

            # Load generated image if exists
            if os.path.isfile(gen_path):
                gen_img = Image.open(gen_path).convert("RGB")
            elif gen_img is None:
                continue

            # CLIP Text-Image
            ti_score = clip_text_image_score(
                clip_model, clip_preprocess, clip_type,
                prompt, gen_img, device
            )
            concept_ti_scores.append(ti_score)

            # CLIP Image-Image (against each reference)
            for ref_img in ref_images[:3]:  # limit refs
                ii_score = clip_image_image_score(
                    clip_model, clip_preprocess, clip_type,
                    gen_img, ref_img, device
                )
                concept_ii_scores.append(ii_score)

        if concept_ti_scores:
            avg_ti = np.mean(concept_ti_scores)
            all_text_image_scores.append(avg_ti)
        if concept_ii_scores:
            avg_ii = np.mean(concept_ii_scores)
            all_image_image_scores.append(avg_ii)

        per_concept_results.append({
            "concept_idx": cidx,
            "class_prompt": class_prompt,
            "num_prompts": len(concept_ti_scores),
            "avg_clip_ti": float(np.mean(concept_ti_scores)) if concept_ti_scores else None,
            "avg_clip_ii": float(np.mean(concept_ii_scores)) if concept_ii_scores else None,
        })

    # Aggregate
    results = {
        "stage": stage_name,
        "agent_config": agent_config_path,
        "num_concepts": len(per_concept_results),
        "clip_text_image": float(np.mean(all_text_image_scores)) if all_text_image_scores else None,
        "clip_image_image": float(np.mean(all_image_image_scores)) if all_image_image_scores else None,
        "per_concept": per_concept_results,
    }

    result_path = os.path.join(output_dir, f"{stage_name}_cc101_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== CC101 Results ({stage_name}) ===")
    print(f"  CLIP Text-Image:  {results['clip_text_image']:.4f}" if results["clip_text_image"] else "  CLIP Text-Image:  N/A")
    print(f"  CLIP Image-Image: {results['clip_image_image']:.4f}" if results["clip_image_image"] else "  CLIP Image-Image: N/A")
    print(f"  Results saved to: {result_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="CC101 T2I Forgetting Evaluation")
    parser.add_argument(
        "--agent_config",
        type=str,
        default=None,
        help="Path to agent config YAML (with experiences). None = no experiences (baseline).",
    )
    parser.add_argument(
        "--cc101_dir",
        type=str,
        default="/data1/data/kangborui/zhongyukun/medmax/customconcept101",
        help="Path to CustomConcept101 dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_dir/cc101_eval",
        help="Output directory for results and generated images",
    )
    parser.add_argument(
        "--stage_name",
        type=str,
        default="baseline",
        help="Stage name for this evaluation (e.g., stage1_scienceqa)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for CLIP")
    parser.add_argument("--max_concepts", type=int, default=None, help="Max concepts to evaluate")
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip image generation, only compute CLIP on existing images",
    )
    args = parser.parse_args()

    run_cc101_evaluation(
        cc101_dir=args.cc101_dir,
        agent_config_path=args.agent_config,
        output_dir=args.output_dir,
        stage_name=args.stage_name,
        device=args.device,
        max_concepts=args.max_concepts,
        skip_generation=args.skip_generation,
    )


if __name__ == "__main__":
    main()

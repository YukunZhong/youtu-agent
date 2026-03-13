import importlib.util
import inspect
import json
from collections import defaultdict
from pathlib import Path

from ...config import EvalConfig
from ...db import EvaluationSample
from ...utils import FileUtils
from .base_llm_processor import BaseLLMJudgeProcesser

VERIFY_DIR = Path(__file__).parent.parent.parent / "practice" / "verify"


class TrainingFreeGRPOProcesser(BaseLLMJudgeProcesser):
    """Processer for training-free GRPO datasets."""

    name = "training_free_grpo"
    config: EvalConfig = None

    def __init__(self, config: EvalConfig) -> None:
        super().__init__(config)
        self.verify_func = self._load_verify_func()
        self.prompts = FileUtils.load_prompts("practice/processor.yaml")

    def preprocess_one(self, sample: EvaluationSample, recorder=None) -> EvaluationSample:
        """Preprocess a single sample with optional experience recorder.

        Args:
            sample: EvaluationSample to preprocess
            recorder: Optional TaskRecorder with experiences

        Returns:
            Updated EvaluationSample
        """
        if recorder is None:
            augmented_question = sample.raw_question
        else:
            curr_experience = recorder.experiences or {}
            formatted_experiences = "\n".join([f"[{i}]. {e}" for i, e in curr_experience.items()])
            augmented_question = FileUtils.get_jinja_template_str(
                self.prompts["PROBLEM_WITH_EXPERIENCE_TEMPLATE"]
            ).render(
                problem=sample.raw_question,
                experiences=formatted_experiences if formatted_experiences else "None",
            )
        # MLLM: inject image-prefix tokens if file_name is a .npy file
        # tokens file contains full MoDE sequence: [BOS, BOI, 1024 img tokens, EOI, text..., EOS]
        # We only inject the image prefix (BOS + BOI + image + EOI = first 1027 tokens)
        # The text portion is already in augmented_question
        if sample.file_name and sample.file_name.endswith(".npy"):
            try:
                import numpy as np

                tokens = np.load(sample.file_name)
                # Extract image prefix: BOS(0) + BOI(8197) + 1024 img tokens + EOI(8196)
                # EOI is at index 1026, so image prefix = tokens[0:1027]
                image_prefix = tokens[:1027].tolist()
                token_str = ",".join(str(int(t)) for t in image_prefix)
                augmented_question = (
                    f"[MODEL_TOKENS]\n{token_str}\n[/MODEL_TOKENS]\n\n"
                    + augmented_question
                )
            except Exception:
                pass

        sample.update(
            augmented_question=augmented_question,
        )
        return sample

    async def judge_one(self, data: EvaluationSample) -> EvaluationSample:
        """Judge a single sample using the loaded verify function."""
        if self.verify_func is None:
            # directly use the default LLM judging method
            return await super().judge_one(data)

        # Check if verify_func is async or sync and call accordingly
        if inspect.iscoroutinefunction(self.verify_func):
            res = await self.verify_func(sample=data, llm=self.judge_client)
        else:
            res = self.verify_func(sample=data, llm=self.judge_client)

        reward = res.get("reward", 0.0)
        reasoning = res.get("reasoning", None)
        data.update(
            judged_response="Correct" if reward == 1.0 else "Incorrect",
            correct=reward == 1.0,
            reward=reward,
            reasoning=reasoning,
        )
        return data

    def calculate_metrics(self, samples: list[EvaluationSample]) -> dict:
        """Calculate metrics from the judged data."""
        all_rewards = []
        problem_to_scores = defaultdict(list)
        num_tool_calls = []
        # calculate tool calls and rewards
        for sample in samples:
            all_rewards.append(sample.reward)
            problem_to_scores[sample.raw_question].append(sample.reward)
            if sample.trajectories:
                num_tool_calls.append(
                    len([each for each in json.loads(sample.trajectories)[0]["trajectory"] if each["role"] == "tool"])
                )
        problem_to_max_score = {problem: max(scores) for problem, scores in problem_to_scores.items()}
        max_K = max((len(scores) for scores in problem_to_scores.values()), default=0)
        stats = {
            f"Mean@{max_K}": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            f"Pass@{max_K}": sum(max_reward for max_reward in problem_to_max_score.values()) / len(problem_to_max_score)
            if problem_to_max_score
            else 0,
            "avg_tool_call": sum(num_tool_calls) / len(num_tool_calls) if num_tool_calls else 0,
        }
        return stats

    def _load_verify_func(self):
        """Load the verification function from the given path."""
        try:
            verify_path = str(VERIFY_DIR / self.config.verify_filename)
            spec = importlib.util.spec_from_file_location("verify_module", verify_path)
            verify_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(verify_module)
            func = getattr(verify_module, self.config.verify_func_name)
        except Exception as e:
            print(
                f"Warning: Failed to load verification function '{self.config.verify_func_name}' from "
                f"'{self.config.verify_filename}': {e}"
            )
            func = None

        return func

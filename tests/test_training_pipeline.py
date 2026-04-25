import json
from pathlib import Path
import unittest

from training.train_trl_colab import build_sft_dataset


class TrainingPipelineTests(unittest.TestCase):
    def test_sft_dataset_spans_multiple_levels_and_seeds(self) -> None:
        dataset = build_sft_dataset(n_episodes=9, seed_start=2000)

        self.assertGreater(len(dataset), 20)
        self.assertTrue({"system", "user", "assistant"}.issubset(dataset[0]))

        levels = set()
        for item in dataset:
            observation = json.loads(item["user"])
            levels.add(observation["committee_handbook"]["chaos_profile"]["active"]["level"])
        self.assertEqual(levels, {1, 2, 3})

    def test_colab_notebook_is_valid_json(self) -> None:
        notebook = json.loads(Path("training/ops_committee_colab.ipynb").read_text(encoding="utf-8"))

        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 6)


if __name__ == "__main__":
    unittest.main()

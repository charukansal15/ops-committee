import unittest

from agents.committee import run_random_committee, run_rule_based_committee


class Phase3EvalPipelineTests(unittest.TestCase):
    def test_rule_based_committee_beats_random_baseline_on_level_three(self) -> None:
        rule = run_rule_based_committee(level=3, seed=101, max_steps=20)
        random = run_random_committee(level=3, seed=101, max_steps=20)

        self.assertEqual(rule["terminal_reason"], "recovered_cleanly")
        self.assertGreater(rule["total_reward"], random["total_reward"])


if __name__ == "__main__":
    unittest.main()

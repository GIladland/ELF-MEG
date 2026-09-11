from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "collect_fmri_long_program_metrics.py"
SPEC = importlib.util.spec_from_file_location("collect_fmri_long_program_metrics", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LongProgramMetricsTest(unittest.TestCase):
    def test_pareto_requires_no_worse_metrics_and_one_strict_gain(self) -> None:
        stronger = {
            "content_overlap_f1": 0.02,
            "word_overlap_f1": 0.08,
            "generated_top5": 0.10,
            "generated_mean_rank": 40.0,
            "bertscore_f1": -0.1,
        }
        weaker = {
            "content_overlap_f1": 0.01,
            "word_overlap_f1": 0.07,
            "generated_top5": 0.08,
            "generated_mean_rank": 45.0,
            "bertscore_f1": -0.2,
        }

        self.assertTrue(MODULE.dominates(stronger, weaker))
        self.assertFalse(MODULE.dominates(weaker, stronger))

    def test_tradeoff_does_not_dominate(self) -> None:
        overlap_model = {
            "content_overlap_f1": 0.03,
            "word_overlap_f1": 0.09,
            "generated_mean_rank": 55.0,
        }
        retrieval_model = {
            "content_overlap_f1": 0.01,
            "word_overlap_f1": 0.06,
            "generated_mean_rank": 35.0,
        }

        self.assertFalse(MODULE.dominates(overlap_model, retrieval_model))
        self.assertFalse(MODULE.dominates(retrieval_model, overlap_model))


if __name__ == "__main__":
    unittest.main()

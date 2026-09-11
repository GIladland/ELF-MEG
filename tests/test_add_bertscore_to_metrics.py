import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "add_bertscore_to_metrics.py"
SPEC = importlib.util.spec_from_file_location("add_bertscore_to_metrics", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RankedExamplesTest(unittest.TestCase):
    def test_ranks_by_f1_stably_and_keeps_source_indices(self):
        ranked = MODULE.ranked_examples(
            ["g0", "g1", "g2"],
            ["t0", "t1", "t2"],
            np.asarray([0.1, 0.2, 0.3]),
            np.asarray([0.4, 0.5, 0.6]),
            np.asarray([0.7, 0.9, 0.9]),
            top_k=2,
        )
        self.assertEqual([row["index"] for row in ranked], [1, 2])
        self.assertEqual(ranked[0]["generated"], "g1")
        self.assertAlmostEqual(ranked[0]["bertscore_f1"], 0.9)


if __name__ == "__main__":
    unittest.main()

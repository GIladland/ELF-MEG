from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "collect_fmri_xattn_results.py"


class CollectFmriCrossAttentionResultsTest(unittest.TestCase):
    def test_ranks_only_complete_validation_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, content, wer, split in (
                ("weak", 0.03, 1.02, "val"),
                ("winner", 0.05, 0.99, "val"),
                ("train_only", 0.80, 0.10, "train"),
            ):
                run = root / name
                run.mkdir()
                (run / "best.pt").touch()
                (run / "best_metrics.json").write_text(
                    json.dumps(
                        {
                            "split": split,
                            "eval_num_examples": 266,
                            "step": 250,
                            "generation_quality": {
                                "content_words_overlap": content,
                                "words_overlap": content + 0.02,
                                "word_error_rate": wer,
                            },
                        }
                    )
                )
            output = root / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--run-glob",
                    str(root / "*"),
                    "--output-json",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text())

        self.assertEqual(report["complete_runs"], 2)
        self.assertEqual(report["leader"]["run"], "winner")
        self.assertTrue(report["leader"]["beats_t5_content"])
        self.assertEqual(report["rejected"][0]["run"], "train_only")


if __name__ == "__main__":
    unittest.main()

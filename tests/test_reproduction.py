"""Checks for scientific result identity and safe reproduction outputs."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recovery_evaluator", ROOT / "experiments" / "evaluate_simulation_recovery.py"
)
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


class RecoveryIntegrityTests(unittest.TestCase):
    def test_permutation_preserves_exact_recovery(self):
        truth = pd.DataFrame({
            "theta_B1": [.8, .1, .2], "theta_B2": [.1, .7, .3],
            "theta_B3": [.1, .2, .5],
        })
        q = pd.DataFrame({"q0": truth.theta_B3, "q1": truth.theta_B1, "q2": truth.theta_B2})
        cols, matched, error = evaluator.best_column_matching(q, truth)
        self.assertEqual(cols, ["q0", "q1", "q2"])
        self.assertEqual(matched, ["theta_B3", "theta_B1", "theta_B2"])
        self.assertEqual(error, 0)

    def test_component_loss_is_not_silently_scored(self):
        q = pd.DataFrame({"q0": [1.], "q1": [0.]})
        truth = pd.DataFrame({"theta_B1": [1.], "theta_B2": [0.], "theta_B3": [0.]})
        with self.assertRaisesRegex(ValueError, "equal estimated and true component counts"):
            evaluator.best_column_matching(q, truth)

    def test_missing_patient_truth_fails_without_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "data").mkdir()
            (base / "step2").mkdir()
            pd.DataFrame({"patient_id": ["P1"], "theta_B1": [1.],
                          "theta_B2": [0.], "theta_B3": [0.]}).to_csv(
                base / "data" / "simulated_patients_truth.csv", index=False
            )
            pd.DataFrame({"patient_num": ["P2"], "q0": [1.], "q1": [0.], "q2": [0.]}).to_csv(
                base / "step2" / "q_vectors_seed_35_em.csv", index=False
            )
            with self.assertRaisesRegex(ValueError, "Missing truth"):
                evaluator.main(base)
            self.assertFalse((base / "evaluation").exists())

    def test_duplicate_patient_estimates_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "data").mkdir()
            (base / "step2").mkdir()
            pd.DataFrame({"patient_id": ["P1"], "theta_B1": [1.],
                          "theta_B2": [0.], "theta_B3": [0.]}).to_csv(
                base / "data" / "simulated_patients_truth.csv", index=False
            )
            pd.DataFrame({"patient_num": ["P1", "P1"], "q0": [1., 1.],
                          "q1": [0., 0.], "q2": [0., 0.]}).to_csv(
                base / "step2" / "q_vectors_seed_35_em.csv", index=False
            )
            with self.assertRaises(pd.errors.MergeError):
                evaluator.main(base)

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("existing result")
            result = subprocess.run([
                sys.executable, "-B", str(ROOT / "experiments" / "reproduce_simulations.py"),
                "--output-dir", directory,
            ], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Output already exists", result.stderr)
            self.assertEqual(sentinel.read_text(), "existing result")
            self.assertEqual(list(Path(directory).iterdir()), [sentinel])


if __name__ == "__main__":
    unittest.main()

"""Regression coverage for R train/evaluation log publication."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from think_bridge.training.lifecycle_log import LifecycleLogger
from think_bridge.training.progress import BridgeLoggingWindow


class TrainingLoggingTest(unittest.TestCase):
    def test_train_losses_reach_console_and_file_once(self):
        metrics = {
            "loss_total": 3.0,
            "loss_ce": 1.0,
            "loss_match": 0.5,
            "loss_specific": 1.5,
            "preclip_grad_R": 2.0,
            "lr_R": 0.0002,
            "update_seconds": 0.5,
            "cuda_allocated_peak_bytes": 0,
            "cuda_reserved_peak_bytes": 0,
            "throughput_samples_per_second": 256,
        }
        window = BridgeLoggingWindow(metric_names=metrics)
        window.add(metrics)
        args = dict(
            route="route1",
            owner="R",
            phase="route1",
            step=1,
            total_steps=10,
            epoch=0.1,
            window_metrics=window.means(),
            elapsed_seconds=0.5,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logging.jsonl"
            with patch("think_bridge.training.lifecycle_log.write_progress") as console:
                self.assertTrue(LifecycleLogger(path).append_train(**args))
                message = console.call_args.args[0]
                for name in ("loss_ce", "loss_match", "loss_specific"):
                    self.assertIn(name, message)
                self.assertFalse(LifecycleLogger(path).append_train(**args))
                self.assertEqual(console.call_count, 1)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["loss"], 3.0)
            for name in ("loss_ce", "loss_match", "loss_specific"):
                self.assertEqual(rows[0][name], metrics[name])

    def test_evaluation_accepts_per_question_donor_counts(self):
        row = dict(
            event="eval",
            stage="Route1",
            step=1,
            epoch=0,
            true_z_full_accuracy=0.5,
            wrong_donor_counts=[2, 1, 0],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logging.jsonl"
            with patch("think_bridge.training.lifecycle_log.write_progress"):
                self.assertTrue(LifecycleLogger(path)._append(row))
                self.assertFalse(LifecycleLogger(path)._append(row))
            self.assertEqual(
                json.loads(path.read_text())["wrong_donor_counts"], [2, 1, 0]
            )


if __name__ == "__main__":
    unittest.main()

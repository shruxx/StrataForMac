"""tools/test_strata_runner.py - Unit tests for macOS Strata engine runner."""
from __future__ import annotations

import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import strata_runner


class TestStrataRunner(unittest.TestCase):
    def test_parse_args_full(self):
        args = [
            "--serve",
            "--pack", "/path/to/pack",
            "--native", "/path/to/model.gguf",
            "--no-ple",
            "--expert-cache", "auto",
            "--prefill", "auto",
            "--max-context", "131072",
            "--kv", "int8",
            "--threads", "8",
        ]
        cfg = strata_runner.parse_args(args)
        self.assertTrue(cfg["serve"])
        self.assertEqual(cfg["model_path"], "/path/to/model.gguf")
        self.assertEqual(cfg["pack_path"], "/path/to/pack")
        self.assertEqual(cfg["max_context"], 131072)
        self.assertEqual(cfg["kv"], "int8")
        self.assertEqual(cfg["threads"], 8)

    def test_parse_args_defaults(self):
        cfg = strata_runner.parse_args(["--serve"])
        self.assertTrue(cfg["serve"])
        self.assertIsNone(cfg["model_path"])
        self.assertEqual(cfg["max_context"], 4096)
        self.assertEqual(cfg["kv"], "int8")

    def test_perf_cores(self):
        cores = strata_runner.get_perf_cores()
        self.assertIsInstance(cores, int)
        self.assertGreaterEqual(cores, 1)

    def test_handle_generation_parsing(self):
        line = "GEN 256 temperature=0.7 top_p=0.95 top_k=40 min_p=0.05 penalty_repeat=1.1 seed=123 100,200,300"
        parts = line.strip().split()
        max_new = int(parts[1])
        sampling = {}
        for item in parts[2:-1]:
            if "=" in item:
                k, _, v = item.partition("=")
                sampling[k] = v
        tokens = [int(t) for t in parts[-1].split(",")]

        self.assertEqual(max_new, 256)
        self.assertEqual(tokens, [100, 200, 300])
        self.assertEqual(sampling["temperature"], "0.7")
        self.assertEqual(sampling["top_p"], "0.95")
        self.assertEqual(sampling["top_k"], "40")
        self.assertEqual(sampling["min_p"], "0.05")
        self.assertEqual(sampling["penalty_repeat"], "1.1")
        self.assertEqual(sampling["seed"], "123")


if __name__ == "__main__":
    unittest.main()

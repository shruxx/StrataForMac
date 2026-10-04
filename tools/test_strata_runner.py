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

    def test_server_cmd_lets_llama_cpp_fit(self):
        # an explicit -ngl switches llama.cpp's --fit off: a model larger than the Metal working set then fails
        cmd = strata_runner.llama_server_cmd("llama-server", "m.gguf", 131072, 8080, 8, "int8")
        self.assertNotIn("-ngl", cmd)
        self.assertNotIn("--n-gpu-layers", cmd)
        self.assertEqual(cmd[cmd.index("--fit") + 1], "on")
        self.assertEqual(cmd[cmd.index("-c") + 1], "131072")
        self.assertEqual(cmd[cmd.index("-ctk") + 1], "q8_0")

    def test_gpu_budget_shared_memory(self):
        G = strata_runner.GIB
        kolibri = 47454113472
        # 48 GB M5 Pro, working set ~36 GiB: the model does not fit, so the GPU gets 60% of what macOS leaves
        self.assertEqual(strata_runner.gpu_budget(48 * G, 36 * G, kolibri), int(40 * G * 0.6))
        self.assertEqual(strata_runner.fit_margin_mib(36 * G, strata_runner.gpu_budget(48 * G, 36 * G, kolibri)), (36 * G - int(40 * G * 0.6)) // 2 ** 20)
        # 64 GB: it fits whole, up to the working set
        self.assertEqual(strata_runner.gpu_budget(64 * G, 48 * G, kolibri), 48 * G)
        # a small model: the whole working set, and llama.cpp's own 1 GiB margin
        self.assertEqual(strata_runner.fit_margin_mib(21 * G, strata_runner.gpu_budget(32 * G, 21 * G, 4 * G)), 1024)
        # GPU and the CPU-side page cache never exceed what is left after macOS's reserve
        for ram in (16, 24, 32, 36, 48, 64, 96, 128):
            b = strata_runner.gpu_budget(ram * G, int(ram * G * 0.75), kolibri)
            self.assertLessEqual(b, ram * G - strata_runner.OS_RESERVE)

    def test_gpu_layers_whole_layers_from_the_end(self):
        # -ngl N: the output head and the last N-1 blocks
        self.assertEqual(strata_runner.gpu_layer_count([10] * 8, 5, 4), 0)
        self.assertEqual(strata_runner.gpu_layer_count([10] * 8, 5, 5), 1)
        self.assertEqual(strata_runner.gpu_layer_count([10] * 8, 5, 34), 3)
        self.assertEqual(strata_runner.gpu_layer_count([10] * 8, 5, 1000), 9)
        cmd = strata_runner.llama_server_cmd("llama-server", "m.gguf", 4096, 8080, 8, "int8", gpu_layers=26)
        self.assertEqual(cmd[cmd.index("--fit") + 1], "off")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "26")
        self.assertNotIn("--fit-target", cmd)

    def test_no_repack_only_when_paging(self):
        G = strata_runner.GIB
        self.assertTrue(strata_runner.pages_from_ssd(48 * G, 47454113472))     # Kolibri Q4_K_M on 48 GB
        self.assertFalse(strata_runner.pages_from_ssd(64 * G, 47454113472))    # fits whole on 64 GB
        cmd = strata_runner.llama_server_cmd("llama-server", "m.gguf", 4096, 8080, 8, "int8", 12288, True)
        self.assertIn("--no-repack", cmd)
        self.assertEqual(cmd[cmd.index("--fit-target") + 1], "12288")
        self.assertNotIn("--no-repack", strata_runner.llama_server_cmd("llama-server", "m.gguf", 4096, 8080, 8, "int8"))

    def test_model_bytes_counts_all_shards(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for i, n in ((1, 10), (2, 20)):
                Path(d, f"m-0000{i}-of-00002.gguf").write_bytes(b"x" * n)
            Path(d, "other.gguf").write_bytes(b"x" * 5)
            self.assertEqual(strata_runner.model_bytes(Path(d, "m-00001-of-00002.gguf")), 30)
            self.assertEqual(strata_runner.model_bytes(Path(d, "other.gguf")), 5)

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

"""Asset schema and path behavior without a simulator installation."""

import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from praxis import config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def stats_file(self, arrays):
        values = np.empty(len(arrays), dtype=object)
        values[:] = arrays
        path = self.root / "stats.npy"
        np.save(path, values)
        return path

    def valid_stats(self):
        return [np.zeros(7), np.ones(7), np.zeros(8), np.ones(8)]

    def test_stats_validates_schema(self):
        values = config.load_stats(self.stats_file(self.valid_stats()))
        self.assertEqual([v.shape for v in values], [(7,), (7,), (8,), (8,)])
        self.assertTrue(all(v.dtype == np.float32 for v in values))

    def test_stats_reject_invalid_values(self):
        for index, value in [
            (0, np.zeros(8)),
            (1, np.zeros(7)),
            (3, -np.ones(8)),
            (0, np.full(7, np.nan)),
            (2, np.full(8, np.inf)),
            (0, np.array(["bad"] * 7)),
        ]:
            with self.subTest(index=index, value=value):
                arrays = self.valid_stats()
                arrays[index] = value
                with self.assertRaisesRegex(ValueError, "Invalid normalization file"):
                    config.load_stats(self.stats_file(arrays))
        with self.assertRaisesRegex(ValueError, "four arrays"):
            config.load_stats(self.stats_file(self.valid_stats()[:3]))

    def test_stats_missing_file_has_context(self):
        with self.assertRaisesRegex(ValueError, "missing.npy"):
            config.load_stats(self.root / "missing.npy")

    def test_resolve_assets_from_another_cwd(self):
        previous = Path.cwd()
        self.addCleanup(os.chdir, previous)
        os.chdir(self.root)
        self.stats_file(self.valid_stats())
        self.assertEqual(config.load_stats("./stats.npy")[0].shape, (7,))
        self.assertEqual(
            config.resolve_asset("normalization.npy"), config.CHECKPOINT_DIR / "normalization.npy"
        )
        self.assertEqual(config.resolve_asset("custom.npy", self.root), self.root / "custom.npy")
        self.assertEqual(config.resolve_asset("./local.npy").resolve(), self.root / "local.npy")
        self.assertEqual(config.resolve_asset("assets/local.npy"), Path("assets/local.npy"))
        self.assertEqual(
            config.resolve_asset(self.root / "absolute.npy"), self.root / "absolute.npy"
        )
        with self.assertRaisesRegex(ValueError, "empty"):
            config.resolve_asset("")

    def test_environment_overrides_checkpoint_root(self):
        try:
            with patch.dict(os.environ, {"PRAXIS_CHECKPOINT_DIR": str(self.root)}):
                importlib.reload(config)
                self.assertEqual(config.CHECKPOINT_DIR, self.root)
                self.assertEqual(config.DEFAULT_STATS, self.root / "normalization.npy")
                self.assertEqual(config.DEFAULT_TASK_MAP, self.root / "task_map.json")
        finally:
            importlib.reload(config)

    def write_map(self, value):
        path = self.root / "map.json"
        path.write_text(json.dumps(value))
        return path

    def test_task_map_valid_schema(self):
        self.assertEqual(
            config.load_task_map(self.write_map({"2": ["libero_goal", 3]})), {2: ("libero_goal", 3)}
        )

    def test_task_map_rejects_invalid_schema(self):
        cases = [
            {},
            [],
            {"-1": ["suite", 0]},
            {"x": ["suite", 0]},
            {"0": ["suite", -1]},
            {"0": ["suite", True]},
            {"0": ["suite", 1.5]},
            {"0": ["", 1]},
            {"0": ["suite"]},
            {"0": ["suite", 0], "00": ["suite", 1]},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid task map"):
                config.load_task_map(self.write_map(value))

    def test_task_map_invalid_json_has_context(self):
        path = self.root / "broken.json"
        path.write_text("{")
        with self.assertRaisesRegex(ValueError, "broken.json"):
            config.load_task_map(path)


if __name__ == "__main__":
    unittest.main()

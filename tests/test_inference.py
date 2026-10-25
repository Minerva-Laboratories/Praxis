"""CPU inference, checkpoint compatibility, validation and evaluation cleanup."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from praxis.cont_infer import ContInfer
from praxis.eval_cont import evaluate, parse_args
from praxis.policy_eff import ContPolicyEff
from praxis.policy_cvae_eff import CVAEEff


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        stats = np.empty(4, dtype=object)
        stats[:] = [np.zeros(7), np.ones(7), np.zeros(8), np.ones(8)]
        cls.stats = cls.root / "normalization.npy"
        np.save(cls.stats, stats)
        cls.cfg = dict(d=32, w=0.5, depth=1, mlp=1, head_depth=1, vision_v2=True, ls_init=None)
        model = ContPolicyEff(n_tasks=2, chunk=2, p_drop=0.0, **cls.cfg)
        cls.checkpoint = dict(
            model=model.state_dict(), n_tasks=2, chunk=2, res=32, policy="eff", eff_cfg=cls.cfg
        )
        cls.path = cls.root / "regression.pt"
        torch.save(cls.checkpoint, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def load(self, **kwargs):
        return ContInfer(self.path, dev="cpu", stats=self.stats, **kwargs)

    def observations(self):
        return np.zeros((1, 2, 32, 32, 3), dtype=np.uint8), np.zeros((1, 8)), [1]

    def test_regression_cpu_is_finite_and_deterministic(self):
        policy = self.load()
        first = policy.act(*self.observations())
        second = policy.act(*self.observations())
        self.assertEqual(first.shape, (1, 2, 7))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(np.all(np.abs(first) <= 1))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(policy.m.state_dict()), set(self.checkpoint["model"]))

    def test_generative_cpu_uses_prior_mean(self):
        cfg = dict(d=32, w=0.5, enc_depth=1, prior_depth=1, dec_depth=1, head_depth=1, ls_init=None)
        model = CVAEEff(n_tasks=2, chunk=2, latent_dim=4, **cfg)
        path = self.root / "generative.pt"
        torch.save(
            dict(
                model=model.state_dict(),
                n_tasks=2,
                chunk=2,
                res=32,
                policy="cvae_eff",
                latent_dim=4,
                eff_cfg=cfg,
            ),
            path,
        )
        policy = ContInfer(path, dev="cpu", stats=self.stats, deterministic=True)
        result = policy.act(*self.observations())
        self.assertEqual(result.shape, (1, 2, 7))
        np.testing.assert_array_equal(result, policy.act(*self.observations()))

    def test_checkpoint_metadata_is_validated(self):
        for change in [
            dict(policy="unknown"),
            dict(n_tasks=0),
            dict(chunk=-1),
            dict(hist=True),
            dict(eff_cfg=[]),
            dict(policy=[]),
        ]:
            with self.subTest(change=change):
                path = self.root / "invalid.pt"
                torch.save({**self.checkpoint, **change}, path)
                with self.assertRaises(ValueError):
                    ContInfer(path, dev="cpu", stats=self.stats)

    def test_corrupt_checkpoint_has_path_context(self):
        path = self.root / "corrupt.pt"
        path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "Cannot read tensor checkpoint.*corrupt.pt"):
            ContInfer(path, dev="cpu", stats=self.stats)

    def test_legacy_checkpoint_without_policy_keeps_original_class(self):
        from praxis.policy_cont import ContPolicy

        model = ContPolicy(n_tasks=2, chunk=2, p_drop=0.0)
        path = self.root / "legacy.pt"
        torch.save(dict(model=model.state_dict(), n_tasks=2, chunk=2, res=32), path)
        policy = ContInfer(path, dev="cpu", stats=self.stats)
        self.assertIsInstance(policy.m, ContPolicy)
        self.assertEqual(policy.act(*self.observations()).shape, (1, 2, 7))

    def test_input_validation(self):
        policy = self.load()
        images, states, tasks = self.observations()
        cases = [
            (images.astype(np.float32), states, tasks),
            (images[:, :1], states, tasks),
            (images, np.zeros((1, 7)), tasks),
            (images, np.full((1, 8), np.nan), tasks),
            (images, states, [-1]),
            (images, states, [2]),
            (images, states, [1.0]),
        ]
        for args in cases:
            with (
                self.subTest(shapes=[np.shape(value) for value in args]),
                self.assertRaises(ValueError),
            ):
                policy.act(*args)

    def test_integer_task_arrays_are_converted_to_embedding_indices(self):
        policy = self.load()
        images, states, _ = self.observations()
        result = policy.act(images, states, np.array([1], dtype=np.uint8))
        self.assertEqual(result.shape, (1, 2, 7))

    def test_temporal_buffers_require_metadata_and_can_reset(self):
        policy = self.load()
        policy.hist = policy.vhist = 2
        policy.vstride = 1
        with self.assertRaisesRegex(ValueError, "metadata"):
            policy.act(*self.observations())
        policy._hist_state(np.zeros(8), 0, 5, 0)
        policy._frame_stack(self.observations()[0][0], 0, 5, 0)
        policy._hist_state(np.ones(8), 1, 6, 0)
        policy.reset(0, 5)
        self.assertNotIn((0, 5), policy.hbuf)
        self.assertNotIn((0, 5), policy.fbuf)
        self.assertIn((1, 6), policy.hbuf)
        policy.reset()
        self.assertFalse(policy.hbuf)
        self.assertFalse(policy.fbuf)
        with self.assertRaises(ValueError):
            policy.reset(task=1)

    def test_inference_does_not_replace_torch_loader(self):
        loader = torch.load
        self.load()
        self.assertIs(torch.load, loader)


class EvaluationTests(unittest.TestCase):
    def test_parser_validates_positive_counts(self):
        for flag, value in [
            ("--episodes", "0"),
            ("--workers", "-1"),
            ("--exec_steps", "0"),
            ("--steps", "-2"),
            ("--init_offset", "-1"),
            ("--temporal_m", "nan"),
        ]:
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--ckpt", "policy.pt", flag, value])
        args = parse_args(["--ckpt", "policy.pt", "--device", "cpu", "--task-map", "custom.json"])
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.task_map, "custom.json")

    def test_help_has_no_simulator_or_torch_import(self):
        code = """import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'robosuite', 'libero'}:
        raise AssertionError('unexpected import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from praxis.eval_cont import parse_args
parse_args(['--help'])
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--device", result.stdout)

    def test_import_does_not_mutate_torch_rng(self):
        code = """import torch
torch.manual_seed(42)
before = torch.get_rng_state().clone()
import praxis.eval_cont
assert torch.equal(before, torch.get_rng_state())
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_workers_closed_when_rollout_raises(self):
        transform = types.ModuleType("robosuite.utils.transform_utils")
        transform.quat2axisangle = lambda value: value[:3]
        dependencies = {name: types.ModuleType(name) for name in ["robosuite", "robosuite.utils"]}
        dependencies["robosuite.utils.transform_utils"] = transform
        with tempfile.TemporaryDirectory() as directory:
            task_map = Path(directory) / "map.json"
            task_map.write_text(json.dumps({"0": ["libero_goal", 0]}))
            with (
                patch.dict(sys.modules, dependencies),
                patch("praxis.cont_infer.ContInfer") as constructor,
                patch("praxis.rollout.SelfExposure") as rollout,
            ):
                policy = constructor.return_value
                policy.n_tasks, policy.chunk, policy.hist, policy.vhist, policy.res = 1, 8, 1, 1, 32
                rollout.return_value.run.side_effect = RuntimeError("worker failure")
                with self.assertRaisesRegex(RuntimeError, "worker failure"):
                    evaluate("unused.pt", task_map=task_map, device="cpu")
                rollout.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

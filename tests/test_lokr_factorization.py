import unittest

import torch
from torch import nn

from modules.module.LoRAModule import LoKrModule


def _make_lokr(base, decompose_factor):
    return LoKrModule(
        prefix="transformer.lokr",
        orig_module=base,
        dim=4,
        alpha=1.0,
        decompose_both=False,
        decompose_factor=decompose_factor,
        use_tucker=False,
        weight_decompose=False,
        dora_on_output=False,
        full_matrix=False,
        train_device=torch.device("cpu"),
    )


class LokrFactorizationTests(unittest.TestCase):
    def test_initialize_weights_raises_for_invalid_explicit_factor(self):
        base = nn.Linear(13, 17, bias=False)
        with self.assertRaises(RuntimeError):
            _make_lokr(base, decompose_factor=5)  # does not divide 13 or 17

    def test_initialize_weights_succeeds_for_auto_factor(self):
        base = nn.Linear(13, 17, bias=False)
        module = _make_lokr(base, decompose_factor=-1)
        self.assertTrue(
            getattr(module, "lokr_w1", None) is not None
            or (getattr(module, "lokr_w1_a", None) is not None and getattr(module, "lokr_w1_b", None) is not None)
        )
        self.assertTrue(
            getattr(module, "lokr_w2", None) is not None
            or (getattr(module, "lokr_w2_a", None) is not None and getattr(module, "lokr_w2_b", None) is not None)
        )


if __name__ == "__main__":
    unittest.main()

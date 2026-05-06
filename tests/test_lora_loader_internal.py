import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from modules.modelLoader.Flux2ModelLoader import Flux2LoRALoader
from modules.modelLoader.ZImageModelLoader import ZImageLoRALoader
from modules.modelLoader.chroma.ChromaLoRALoader import ChromaLoRALoader
from modules.modelLoader.flux.FluxLoRALoader import FluxLoRALoader
from modules.modelLoader.hiDream.HiDreamLoRALoader import HiDreamLoRALoader
from modules.modelLoader.hunyuanVideo.HunyuanVideoLoRALoader import HunyuanVideoLoRALoader
from modules.modelLoader.stableDiffusion3.StableDiffusion3LoRALoader import StableDiffusion3LoRALoader
from modules.modelLoader.pixartAlpha.PixArtAlphaLoRALoader import PixArtAlphaLoRALoader
from modules.modelLoader.qwen.QwenLoRALoader import QwenLoRALoader
from modules.modelLoader.sana.SanaLoRALoader import SanaLoRALoader
from modules.modelLoader.stableDiffusion.StableDiffusionLoRALoader import StableDiffusionLoRALoader
from modules.modelLoader.stableDiffusionXL.StableDiffusionXLLoRALoader import StableDiffusionXLLoRALoader
from modules.modelLoader.mixin.LoRALoaderMixin import LoRALoaderMixin
from modules.modelLoader.wuerstchen.WuerstchenLoRALoader import WuerstchenLoRALoader
from modules.util.ModelNames import ModelNames
from modules.util.convert.lora.convert_lora_util import LoraConversionKeySet
from modules.util.convert.lora.convert_flux2_lora import convert_flux2_lora_key_sets


class _TestInternalLoader(LoRALoaderMixin):
    def _get_convert_key_sets(self, model):
        return convert_flux2_lora_key_sets()

    def load(self, model, model_names: ModelNames):
        return self._load(model, model_names)


class _TestInternalLoaderLegacyConversion(LoRALoaderMixin):
    def _get_convert_key_sets(self, model):
        return [LoraConversionKeySet("unet", "lora_unet")]

    def load(self, model, model_names: ModelNames):
        return self._load(model, model_names)


class LoRALoaderInternalTests(unittest.TestCase):
    def _load_from_internal_backup(
        self,
        state_dict: dict[str, torch.Tensor],
        loader: LoRALoaderMixin,
        model: SimpleNamespace | None = None,
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            backup_dir = Path(tmpdir)
            (backup_dir / "meta.json").write_text("{}", encoding="utf-8")
            (backup_dir / "lora").mkdir()
            save_file(state_dict, str(backup_dir / "lora" / "lora.safetensors"))

            model = model or SimpleNamespace(lora_state_dict=None)
            loader.load(model, ModelNames(lora=str(backup_dir)))
            self.assertIsNotNone(model.lora_state_dict)
            return model.lora_state_dict

    def test_internal_loader_keeps_internal_keys_unchanged(self):
        original_state = {
            # Intentionally not part of current Flux2 conversion keysets.
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((4, 4)),
            "transformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 4)),
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        }
        loaded_state_dict = self._load_from_internal_backup(original_state, _TestInternalLoader())
        self.assertEqual(set(loaded_state_dict.keys()), set(original_state.keys()))

    def test_internal_loader_repairs_malformed_transformer_prefix(self):
        malformed_state = {
            "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.lora_down.weight": torch.zeros((4, 4)),
            "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.lora_up.weight": torch.zeros((4, 4)),
            "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(1.0),
        }
        loaded_state_dict = self._load_from_internal_backup(malformed_state, _TestInternalLoader())
        self.assertIn(
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
            loaded_state_dict,
        )
        self.assertNotIn(
            "diffusion_model.txt_intransformer.transformer_blocks.0.attn.to_q.lora_down.weight",
            loaded_state_dict,
        )

    def test_internal_loader_converts_legacy_internal_keys_when_needed(self):
        legacy_state = {
            "unet.input_blocks.4.1.proj_in.lora_down.weight": torch.zeros((4, 4)),
            "unet.input_blocks.4.1.proj_in.lora_up.weight": torch.zeros((4, 4)),
            "unet.input_blocks.4.1.proj_in.alpha": torch.tensor(1.0),
            "custom.unknown_key": torch.tensor(1.0),
        }
        loaded_state_dict = self._load_from_internal_backup(legacy_state, _TestInternalLoaderLegacyConversion())
        self.assertIn("lora_unet.input_blocks.4.1.proj_in.lora_down.weight", loaded_state_dict)
        self.assertIn("custom.unknown_key", loaded_state_dict)

    def test_internal_loader_flux2_converts_legacy_internal_keys(self):
        legacy_state = {
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_down.weight": torch.zeros((4, 4)),
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_up.weight": torch.zeros((4, 4)),
            "diffusion_model.double_blocks.0.img_attn.qkv.alpha": torch.tensor(1.0),
            "custom.keep_me": torch.tensor(1.0),
        }
        loaded_state_dict = self._load_from_internal_backup(legacy_state, Flux2LoRALoader())
        self.assertIn(
            "transformer.transformer_blocks.0.img_attn.qkv.lora_down.weight",
            loaded_state_dict,
        )
        self.assertIn("custom.keep_me", loaded_state_dict)

    def test_internal_loader_sd3_converts_legacy_internal_keys(self):
        legacy_state = {
            "transformer.joint_blocks.0.x_block.attn.qkv.0.lora_down.weight": torch.zeros((4, 4)),
            "transformer.joint_blocks.0.x_block.attn.qkv.0.lora_up.weight": torch.zeros((4, 4)),
            "transformer.joint_blocks.0.x_block.attn.qkv.0.alpha": torch.tensor(1.0),
            "custom.keep_me": torch.tensor(1.0),
        }
        loaded_state_dict = self._load_from_internal_backup(legacy_state, StableDiffusion3LoRALoader())
        self.assertIn(
            "lora_transformer.transformer_blocks.0.attn.to_q.lora_down.weight",
            loaded_state_dict,
        )
        self.assertIn("custom.keep_me", loaded_state_dict)

    def test_internal_loader_supported_models_smoke(self):
        def _simple_model():
            return SimpleNamespace(lora_state_dict=None)

        test_matrix = [
            ("stable-diffusion", StableDiffusionLoRALoader(), _simple_model()),
            ("stable-diffusion-xl", StableDiffusionXLLoRALoader(), _simple_model()),
            ("stable-diffusion-3", StableDiffusion3LoRALoader(), _simple_model()),
            ("flux", FluxLoRALoader(), _simple_model()),
            ("flux-2", Flux2LoRALoader(), _simple_model()),
            ("chroma", ChromaLoRALoader(), _simple_model()),
            ("hunyuan-video", HunyuanVideoLoRALoader(), _simple_model()),
            ("pixart-alpha", PixArtAlphaLoRALoader(), _simple_model()),
            ("qwen", QwenLoRALoader(), _simple_model()),
            ("sana", SanaLoRALoader(), _simple_model()),
            ("hidream", HiDreamLoRALoader(), _simple_model()),
            ("z-image", ZImageLoRALoader(), _simple_model()),
            (
                "wuerstchen-stable-cascade",
                WuerstchenLoRALoader(),
                SimpleNamespace(lora_state_dict=None, model_type=SimpleNamespace(is_stable_cascade=lambda: True)),
            ),
            (
                "wuerstchen-non-stable-cascade",
                WuerstchenLoRALoader(),
                SimpleNamespace(lora_state_dict=None, model_type=SimpleNamespace(is_stable_cascade=lambda: False)),
            ),
        ]

        for model_name, loader, model in test_matrix:
            with self.subTest(model=model_name):
                key_sets = loader._get_convert_key_sets(model)
                if not key_sets:
                    # By design, non-stable-cascade Wuerstchen has no conversion keyset.
                    self.assertEqual(model_name, "wuerstchen-non-stable-cascade")
                    continue

                key_set = next(
                    ks
                    for ks in key_sets
                    if ks.omi_prefix and ks.diffusers_prefix and ks.omi_prefix != "bundle_emb"
                )
                legacy_state = {
                    f"{key_set.omi_prefix}.lora_down.weight": torch.zeros((4, 4)),
                    f"{key_set.omi_prefix}.lora_up.weight": torch.zeros((4, 4)),
                    f"{key_set.omi_prefix}.alpha": torch.tensor(1.0),
                    "custom.keep_me": torch.tensor(1.0),
                }
                loaded_state_dict = self._load_from_internal_backup(legacy_state, loader, model=model)

                self.assertIn(f"{key_set.diffusers_prefix}.lora_down.weight", loaded_state_dict)
                self.assertIn("custom.keep_me", loaded_state_dict)


if __name__ == "__main__":
    unittest.main()

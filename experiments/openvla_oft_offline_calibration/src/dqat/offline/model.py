"""OpenVLA-OFT split into (prefix -> 32 streamed decoder layers -> head) for offline analysis.

The prefix and head are the upstream modules and functions themselves (vision backbone, projector,
proprio projector, `predict_action` up to the first decoder layer, final norm, L1 action head and
un-normalization). Only the decoder layers are run by our own loop, one layer at a time, with the
arguments upstream passes to every layer (causal SDPA attention, positions 0..T-1, no cache).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from dqat.offline.remote import RemoteSafetensors

ROOT = Path(__file__).resolve().parents[3]
REPO = "moojink/openvla-7b-oft-finetuned-libero-10"
REVISION = "95220f9a3421a7ff12d4218e73d09ade830fa9a3"
UNNORM_KEY = "libero_10_no_noops"
NUM_LAYERS = 32
LINEARS = (
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
    "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
)


def upstream():
    """Import upstream OpenVLA-OFT (LIBERO constants) and return the modules we use."""
    path = str(ROOT / "third_party/openvla-oft")
    if path not in sys.path:
        sys.path.insert(0, path)
    from experiments.robot import openvla_utils
    from prismatic.extern.hf import configuration_prismatic, modeling_prismatic, processing_prismatic
    from prismatic.models import action_heads, projectors
    from prismatic.vla import constants

    if (constants.NUM_ACTIONS_CHUNK, constants.ACTION_DIM, constants.PROPRIO_DIM) != (8, 7, 8):
        raise RuntimeError("upstream did not select the LIBERO constants")
    return openvla_utils, configuration_prismatic, modeling_prismatic, processing_prismatic, action_heads, projectors


def component_state_dict(path: Path) -> dict:
    """Upstream load_component_state_dict (strip DDP "module."), mapped to CPU."""
    state = torch.load(path, weights_only=True, map_location="cpu")
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}


class Captured(Exception):
    pass


class OFTEnds:
    def __init__(self, ckpt_dir: Path, remote: RemoteSafetensors, device: str = "mps", dtype=torch.float16):
        utils, cfg_mod, model_mod, proc_mod, heads, projectors = upstream()
        self.utils = utils
        self.device, self.dtype = device, dtype
        config = cfg_mod.OpenVLAConfig.from_pretrained(ckpt_dir)
        config.text_config._attn_implementation = "sdpa"
        config._attn_implementation = "sdpa"
        from accelerate import init_empty_weights

        with init_empty_weights():  # parameters on "meta"; nothing is allocated for the 7B decoder
            model = model_mod.OpenVLAForActionPrediction(config)
        model.vision_backbone.set_num_images_in_input(2)
        keep = {
            "vision_backbone": model.vision_backbone,
            "projector": model.projector,
            "language_model.model.embed_tokens": model.language_model.model.embed_tokens,
            "language_model.model.norm": model.language_model.model.norm,
        }
        cache = ckpt_dir / "ends.pt"  # ~1.7 GB of non-decoder weights, fetched once
        if cache.exists():
            all_tensors = torch.load(cache)
        else:
            all_tensors = remote.get([k for k in remote.weight_map if any(k.startswith(p + ".") for p in keep)])
            torch.save(all_tensors, cache)
        for prefix, module in keep.items():
            tensors = {k: v for k, v in all_tensors.items() if k.startswith(prefix + ".")}
            module.load_state_dict({k[len(prefix) + 1 :]: v for k, v in tensors.items()}, strict=True, assign=True)
            module.to(device, dtype)
        for name, param in list(model.vision_backbone.named_parameters()) + list(model.projector.named_parameters()):
            if param.is_meta:
                raise RuntimeError(f"{name} was not loaded")
        model.eval()
        model.norm_stats = json.loads((ckpt_dir / "dataset_statistics.json").read_text())
        self.model = model
        from transformers import AutoTokenizer

        # Same components AutoProcessor builds from this checkpoint, without remote-code loading.
        self.processor = proc_mod.PrismaticProcessor(
            image_processor=proc_mod.PrismaticImageProcessor.from_pretrained(ckpt_dir),
            tokenizer=AutoTokenizer.from_pretrained(ckpt_dir),
        )
        self.action_head = heads.L1RegressionActionHead(input_dim=4096, hidden_dim=4096, action_dim=7)
        self.action_head.load_state_dict(component_state_dict(ckpt_dir / "action_head--150000_checkpoint.pt"))
        self.action_head.to(device, dtype).eval()
        self.proprio_projector = projectors.ProprioProjector(llm_dim=4096, proprio_dim=8)
        self.proprio_projector.load_state_dict(
            component_state_dict(ckpt_dir / "proprio_projector--150000_checkpoint.pt")
        )
        self.proprio_projector.to(device, dtype).eval()
        self.num_patches = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input() + 1

    # ---- observation -> model inputs (upstream get_vla_action, minus the model call) -------------

    def prepare(self, full_image: np.ndarray, wrist_image: np.ndarray, state: np.ndarray, instruction: str):
        """Images in robosuite convention (as obs["agentview_image"]); state = 8D eef/gripper."""
        utils = self.utils
        img = utils.resize_image_for_policy(np.ascontiguousarray(full_image[::-1, ::-1]), 224)
        wrist = utils.resize_image_for_policy(np.ascontiguousarray(wrist_image[::-1, ::-1]), 224)

        class Cfg:
            center_crop = True

        primary, wrist_pil = utils.prepare_images_for_vla([img, wrist], Cfg)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        inputs = self.processor(prompt, primary)
        wrist_inputs = self.processor(prompt, wrist_pil)
        pixel_values = torch.cat([inputs["pixel_values"], wrist_inputs["pixel_values"]], dim=1)
        proprio = utils.normalize_proprio(np.asarray(state, dtype=np.float64), self.model.norm_stats[UNNORM_KEY]["proprio"])
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": pixel_values,
            "proprio": np.asarray(proprio),
            "primary_image": np.asarray(primary),
            "wrist_image": np.asarray(wrist_pil),
        }

    @torch.no_grad()
    def layer0_input(self, prepared: dict) -> tuple[torch.Tensor, int]:
        """Run upstream predict_action until decoder layer 0; return its input and the action start."""
        layers = self.model.language_model.model.layers
        captured = {}

        def hook(module, args):
            captured["h"] = args[0].detach()
            raise Captured

        handle = layers[0].register_forward_pre_hook(hook)
        input_ids = prepared["input_ids"].to(self.device)
        try:
            self.model.predict_action(
                input_ids=input_ids,
                attention_mask=prepared["attention_mask"].to(self.device),
                pixel_values=prepared["pixel_values"].to(self.device, self.dtype),
                unnorm_key=UNNORM_KEY,
                do_sample=False,
                proprio=prepared["proprio"],
                proprio_projector=self.proprio_projector,
                noisy_action_projector=None,
                action_head=self.action_head,
                use_film=False,
            )
        except Captured:
            pass
        finally:
            handle.remove()
        h = captured["h"]
        # Upstream: append the empty token 29871 if absent; NUM_PROMPT_TOKENS = len(input_ids) - 1.
        n_ids = input_ids.shape[-1] + (0 if bool(torch.all(input_ids[:, -1] == 29871)) else 1)
        action_start = self.num_patches + n_ids - 1
        if h.shape[1] != self.num_patches + n_ids + 7 * 8 + 1:
            raise RuntimeError(f"unexpected multimodal length {h.shape[1]}")
        return h[0], action_start

    @torch.no_grad()
    def head(self, last_hidden: torch.Tensor, action_start: int) -> tuple[np.ndarray, np.ndarray]:
        """Final norm + upstream L1 head + un-normalization. Returns (normalized, actions), both (8, 7)."""
        h = self.model.language_model.model.norm(last_hidden.to(self.device, self.dtype)[None])
        acts = h[:, action_start : action_start + 56, :]
        normalized = self.action_head.predict_action(acts).reshape(8, 7).float().cpu().numpy()
        return normalized, np.asarray(self.model._unnormalize_actions(normalized, UNNORM_KEY))


def layer_names(index: int) -> list[str]:
    prefix = f"language_model.model.layers.{index}."
    return [
        prefix + n
        for n in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            *[f"{name}.weight" for name in LINEARS],
        )
    ]


def build_layer(text_config, index: int, tensors: dict, device, dtype):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer

    text_config._attn_implementation = "sdpa"
    layer = LlamaDecoderLayer(text_config, index)
    prefix = f"language_model.model.layers.{index}."
    state = {k[len(prefix) :]: v for k, v in tensors.items()}
    missing, unexpected = layer.load_state_dict(state, strict=False)
    if unexpected or [m for m in missing if not m.endswith("inv_freq")]:
        raise RuntimeError(f"layer {index}: missing {missing}, unexpected {unexpected}")
    return layer.to(device, dtype).eval()


def run_layer(layer, hidden: torch.Tensor) -> torch.Tensor:
    """Exactly the call upstream's LlamaModel makes (no padding: SDPA causal, no cache)."""
    t = hidden.shape[1]
    positions = torch.arange(t, device=hidden.device)
    return layer(
        hidden,
        attention_mask=None,
        position_ids=positions[None].expand(hidden.shape[0], -1),
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=positions,
    )[0]

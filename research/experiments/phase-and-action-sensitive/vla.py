"""Differentiable OpenVLA-OFT forward on a demo frame, matching the evaluation preprocessing."""

import torch


def action_hidden(policy, full, wrist, state, instruction):
    """Action-token hidden states (1, 56, 4096) for 256px agentview/wrist images and 8D state."""
    from experiments.robot.openvla_utils import normalize_proprio, prepare_images_for_vla

    model, cfg = policy.model, policy.cfg
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    images = prepare_images_for_vla([full, wrist], cfg)
    inputs = policy.processor(prompt, images[0]).to("cuda", dtype=torch.bfloat16)
    wrist_pixels = policy.processor(prompt, images[1]).to("cuda", dtype=torch.bfloat16)["pixel_values"]
    inputs["pixel_values"] = torch.cat([inputs["pixel_values"], wrist_pixels], dim=1)
    proprio = normalize_proprio(state, model.norm_stats[cfg.unnorm_key]["proprio"])
    _, hidden = model.predict_action(
        **inputs, unnorm_key=cfg.unnorm_key, do_sample=False, proprio=proprio,
        proprio_projector=policy.proprio_projector, action_head=policy.action_head, use_film=False,
    )
    return hidden

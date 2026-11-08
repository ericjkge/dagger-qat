"""Serve the unquantized OpenPI PyTorch pi05_libero policy on a local websocket."""

import argparse
import dataclasses
import hashlib
import json
import logging
import os
import random
from pathlib import Path

from vlacal.policy.checkpoint import validate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--denoising-steps", type=int, default=10)
    quant = parser.add_argument_group("quantization (omit --quant-config for the FP policy)")
    quant.add_argument("--quant-config", type=Path, help="JSON file with QuantConfig fields")
    quant.add_argument("--calib", action="append", default=[], help="episode .npz glob(s)")
    quant.add_argument("--calib-per-episode", type=int, default=4)
    quant.add_argument("--calib-max-frames", type=int, default=256)
    quant.add_argument("--calib-seed", type=int, default=0)
    quant.add_argument("--quant-report", type=Path, help="write the per-layer report here")
    args = parser.parse_args()
    validate_checkpoint(args.checkpoint)
    if args.denoising_steps < 1:
        parser.error("--denoising-steps must be positive")
    os.environ["JAX_PLATFORMS"] = "cpu"
    import numpy as np
    import torch
    from openpi.policies import policy_config
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training import config

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Run inside a GPU compute allocation.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    train_config = config.get_config("pi05_libero")
    # Eager inference avoids a long compilation warm-up for a single episode.
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
    )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": args.denoising_steps},
    )
    if not policy._is_pytorch_model:
        raise RuntimeError("Refusing to serve a JAX policy in the PyTorch smoke test")
    metadata = {
        "model": "pi05_libero",
        "backend": "pytorch",
        "quantization": None,
        "precision": "OpenPI BF16/FP32",
        "seed": args.seed,
        "denoising_steps": args.denoising_steps,
        "action_horizon": train_config.model.action_horizon,
        "action_dim": 7,
        "checkpoint": str(args.checkpoint.resolve()),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    if args.quant_config is not None:
        metadata["quantization"] = quantize_policy(policy, args)
        metadata["precision"] = "fake-quantized (see quantization)"
    manifest = args.checkpoint / "vlacal_checkpoint.json"
    if manifest.is_file():
        metadata["checkpoint_provenance"] = json.loads(manifest.read_text())
    logging.info("Policy ready: %s", json.dumps(metadata))
    with torch.inference_mode():
        WebsocketPolicyServer(
            policy, host=args.host, port=args.port, metadata=metadata
        ).serve_forever()


def quantize_policy(policy, args) -> dict:
    """Calibrate on recorded frames (if the config needs data) and fake-quantize in place."""
    from vlacal.calib.frames import calibration_runner, episode_files, load_frames
    from vlacal.quant import QuantConfig, quantize_model

    cfg = QuantConfig.from_json(args.quant_config.read_text())
    calibration = None
    run = None
    if cfg.needs_calibration:
        if not args.calib:
            raise SystemExit("this quantization config needs --calib episode files")
        frames = load_frames(
            args.calib,
            per_episode=args.calib_per_episode,
            max_frames=args.calib_max_frames,
            seed=args.calib_seed,
        )
        files = episode_files(args.calib)
        calibration = {
            "patterns": args.calib,
            "episodes": len(files),
            "frames": len(frames),
            "per_episode": args.calib_per_episode,
            "max_frames": args.calib_max_frames,
            "seed": args.calib_seed,
            "file_list_sha256": hashlib.sha256("\n".join(files).encode()).hexdigest(),
        }
        run = calibration_runner(policy, frames)
    handle = quantize_model(policy._model, cfg, run, log=logging.info)
    if args.quant_report is not None:
        args.quant_report.write_text(json.dumps({**handle.report, "calibration": calibration}))
    return {
        "config": cfg.to_dict(),
        "calibration": calibration,
        "layers": handle.report["n_layers"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

"""Offline kill-check on a GPU node: quantize π0.5 per arm, compare actions to FP on recorded
frames. Run with the model env:  .venv-model/bin/python scripts/offline_gap.py --plan ... --out ...

See configs/killcheck/camera.json for a plan and docs/oracle-experiment.md for the protocol.
"""

import argparse
import dataclasses
import json
import logging
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--denoising-steps", type=int, default=10)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    os.environ["JAX_PLATFORMS"] = "cpu"
    # Plans refer to ${VLACAL_DATA_DIR}; default it like scripts/common.sh does.
    os.environ.setdefault("VLACAL_DATA_DIR", str(Path(__file__).resolve().parents[1] / "data"))
    import numpy as np
    import torch
    from openpi.policies import policy_config
    from openpi.training import config

    from vlacal.eval.offline import Plan, format_summary, run_plan
    from vlacal.policy.checkpoint import validate_checkpoint

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    validate_checkpoint(args.checkpoint)
    plan = Plan.load(args.plan)
    (args.out / "plan.json").write_text(args.plan.read_text())
    train_config = config.get_config("pi05_libero")
    train_config = dataclasses.replace(
        train_config, model=dataclasses.replace(train_config.model, pytorch_compile_mode=None)
    )
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": args.denoising_steps},
    )
    results = run_plan(policy, plan, log=logging.info)
    actions = results.pop("actions")
    np.savez_compressed(
        args.out / "actions.npz",
        **{f"{arm}__{ev}": a for arm, per in actions.items() for ev, a in per.items()},
    )
    results["meta"] = {"gpu": torch.cuda.get_device_name(0), "checkpoint": str(args.checkpoint)}
    (args.out / "results.json").write_text(json.dumps(results, indent=1))
    print(format_summary(results["summary"], plan.reference))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()

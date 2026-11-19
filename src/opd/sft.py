"""Step 2: verl's SFT trainer under configs/sft.yaml, on the episodes opd.collect wrote (opd.sft_data)."""

import hydra
from omegaconf import DictConfig
from verl.trainer.sft_trainer import run_sft
from verl.utils.device import auto_set_device

from opd import CONFIGS


@hydra.main(config_path=CONFIGS, config_name="sft", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    run_sft(config)


if __name__ == "__main__":
    main()

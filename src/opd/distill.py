"""Step 3: PEAR's training run with verl's on-policy distillation, under configs/distill.yaml."""

import hydra
from omegaconf import DictConfig
from pear.trainer.main import Training

from opd import CONFIGS


@hydra.main(config_path=CONFIGS, config_name="distill", version_base=None)
def main(config: DictConfig) -> None:
    Training(config).run()


if __name__ == "__main__":
    main()

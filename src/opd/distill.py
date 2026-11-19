"""Step 3: PEAR's training run with verl's on-policy distillation, under configs/distill.yaml."""

import hydra
from omegaconf import DictConfig
from pear.trainer.main import Training

from opd import CONFIGS, use_chat_template_file


@hydra.main(config_path=CONFIGS, config_name="distill", version_base=None)
def main(config: DictConfig) -> None:
    use_chat_template_file(config)
    Training(config).run()


if __name__ == "__main__":
    main()

"""Step 3: PEAR's training run with verl's on-policy distillation, with configs/step/distill.yaml."""

import hydra
from omegaconf import DictConfig
from pear.trainer.main import Training

from opd import PEAR_CONFIGS, use_chat_template_file


@hydra.main(config_path=PEAR_CONFIGS, config_name="training", version_base=None)
def main(config: DictConfig) -> None:
    use_chat_template_file(config)
    Training(config).run()


if __name__ == "__main__":
    main()

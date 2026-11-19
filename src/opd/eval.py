"""Step 4: PEAR's embodied evaluation under configs/eval.yaml."""

import hydra
from omegaconf import DictConfig
from pear.eval.embodied import EmbodiedEvaluation

from opd import CONFIGS, use_chat_template_file


@hydra.main(config_path=CONFIGS, config_name="eval", version_base=None)
def main(config: DictConfig) -> None:
    use_chat_template_file(config)
    EmbodiedEvaluation(config).run()


if __name__ == "__main__":
    main()

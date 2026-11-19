"""Step 4: PEAR's embodied evaluation with configs/step/eval.yaml."""

import hydra
from omegaconf import DictConfig
from pear.eval.embodied import EmbodiedEvaluation

from opd import PEAR_CONFIGS, use_chat_template_file


@hydra.main(config_path=PEAR_CONFIGS, config_name="embodied", version_base=None)
def main(config: DictConfig) -> None:
    use_chat_template_file(config)
    EmbodiedEvaluation(config).run()


if __name__ == "__main__":
    main()

"""On-policy distillation of PEAR's robot-use reasoning model into smaller students.

Each step's entry point runs on the upstream config it extends (PEAR's embodied evaluation or training, verl's SFT
trainer); the step's own settings are an overlay, configs/step/<step>.yaml, which the scripts select with
--config-dir configs +step=<step>."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"
PEAR_CONFIGS = str(ROOT / "third_party" / "pear" / "pear" / "configs")
VERL_CONFIGS = str(ROOT / "third_party" / "pear" / "3rdparty" / "verl" / "src" / "verl" / "trainer" / "config")


def use_chat_template_file(config) -> None:
    """Sets the policy's chat template from the config's chat_template_file, when a model file names one: the students
    render their prompts with the teacher's template, whose system prompt carries the reasoning effort."""
    path = config.get("chat_template_file")
    if path:
        config.actor_rollout_ref.model.custom_chat_template = Path(path).read_text()

"""On-policy distillation of PEAR's robot-use reasoning model into smaller students."""

from pathlib import Path

CONFIGS = str(Path(__file__).resolve().parents[2] / "configs")


def use_chat_template_file(config) -> None:
    """Sets the policy's chat template from the config's chat_template_file, when a model file names one: the students
    render their prompts with the teacher's template, so both see each episode in the same layout."""
    path = config.get("chat_template_file")
    if path:
        config.actor_rollout_ref.model.custom_chat_template = Path(path).read_text()

"""opd's shared helpers."""

from pathlib import Path

from omegaconf import OmegaConf

from opd import CONFIGS, PEAR_CONFIGS, VERL_CONFIGS, use_chat_template_file


def test_upstream_configs_exist():
    # the entry points' config_path: the step overlays, PEAR's configs, and verl's in PEAR's checkout of it
    for name in ["eval", "collect", "sft", "distill"]:
        assert (CONFIGS / "step" / f"{name}.yaml").is_file()
    upstream = [(PEAR_CONFIGS, "embodied"), (PEAR_CONFIGS, "training"), (VERL_CONFIGS, "sft_trainer_engine")]
    for config_dir, name in upstream:
        assert (Path(config_dir) / f"{name}.yaml").is_file()


def test_chat_template_file_sets_the_policys_template(tmp_path):
    path = tmp_path / "chat_template.jinja"
    path.write_text("{{ messages }}")
    config = OmegaConf.create({"chat_template_file": str(path), "actor_rollout_ref": {"model": {}}})
    use_chat_template_file(config)
    assert config.actor_rollout_ref.model.custom_chat_template == "{{ messages }}"


def test_no_chat_template_file_keeps_the_models_own():
    config = OmegaConf.create({"chat_template_file": None, "actor_rollout_ref": {"model": {}}})
    use_chat_template_file(config)
    assert "custom_chat_template" not in config.actor_rollout_ref.model

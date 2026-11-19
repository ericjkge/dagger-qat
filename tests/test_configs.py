"""Every step's config, composed as its entry point composes it: the upstream config (PEAR's or verl's) with the step's
overlay and the exact overrides its script passes. Hydra refuses a key that does not exist (without +) or a value of
the wrong type, so this checks every name and value the scripts and configs/step/*.yaml set."""

import pytest
from conftest import PEAR, ROOT, VERL_SRC
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_scripts import DISTILL, STEPS

VERL_CONFIGS = VERL_SRC / "verl" / "trainer" / "config"
PEAR_CONFIGS = PEAR / "pear" / "configs"
# the entry points' upstream configs
UPSTREAM = {
    "eval.sh": (PEAR_CONFIGS, "embodied"),
    "collect.sh": (PEAR_CONFIGS, "embodied"),
    "distill.sh": (PEAR_CONFIGS, "training"),
    "sft.sh": (VERL_CONFIGS, "sft_trainer_engine"),
}
# PEAR's configs search verl's and PEAR's model families' packages (pkg://), here their directories, since composing
# needs the YAML files only; the scripts' --config-dir adds configs/ for the step overlays
SEARCHPATH = [f"file://{VERL_CONFIGS}", f"file://{PEAR / 'pear' / 'models'}", f"file://{ROOT / 'configs'}"]


def compose_step(script: str, args: list[str]):
    config_dir, config_name = UPSTREAM[script]
    overrides = [arg for arg in args if not arg.startswith("hydra.")]
    overrides.append(f"hydra.searchpath=[{','.join(SEARCHPATH)}]")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name, overrides=overrides)


@pytest.mark.parametrize("gpu", ["generic", "h100"])
@pytest.mark.parametrize("step", STEPS)
def test_step_config_composes(run_step, step, gpu):
    _, args = run_step(step, GPU=gpu, **STEPS[step])
    config = compose_step(step, args)
    assert config.actor_rollout_ref.model.path == STEPS[step]["MODEL"]
    OmegaConf.to_container(config)  # every value of the composed config is well-formed


def test_eval_and_collect_run_one_pass(run_step, tmp_path):
    for step in ["eval.sh", "collect.sh"]:
        _, args = run_step(step, **STEPS[step])
        config = compose_step(step, args)
        assert config.trainer.total_training_steps == 1
        assert config.trainer.eval_step == 0
    assert config.collect.out_dir == str(tmp_path / "run" / "episodes")


def test_distillation_is_reverse_kl_without_task_reward(run_step):
    _, args = run_step("distill.sh", **DISTILL)
    config = compose_step("distill.sh", args)
    loss = config.distillation.distillation_loss
    assert config.distillation.enabled
    assert (loss.loss_mode, loss.use_policy_gradient, loss.policy_loss_mode, loss.use_task_rewards) == (
        "k1",
        True,
        "vanilla",
        False,
    )
    assert config.distillation.teacher_models.teacher_model.model_path == "/models/teacher"
    # RL-only machinery off: no group filter (no reward variance without a task reward), no reference KL
    assert not config.algorithm.filter_groups.enable
    assert not config.actor_rollout_ref.actor.use_kl_loss
    assert config.actor_rollout_ref.rollout.n == 1
    assert config.trainer.test_freq == 10


def test_sft_config_composes(run_step, tmp_path):
    _, args = run_step("sft.sh", MODEL="/models/student", EPISODES_DIR=tmp_path / "ep", NGPUS=1)
    config = compose_step("sft.sh", args)
    assert config.data.custom_cls.path == "pkg://opd.sft_data"
    assert config.data.custom_cls.name == "EpisodeDataset"
    assert config.data.pad_mode == "no_padding"
    assert config.model.path == "/models/student"

"""The step scripts: the Hydra overrides each one composes, for every step, hardware profile and benchmark."""

import subprocess
import urllib.request
from pathlib import Path

import pytest
from conftest import PEAR, ROOT

MODELS = ROOT / "configs" / "models"
STUDENT = {"MODEL_FILE": MODELS / "qwen3_5_9b.sh", "MODEL": "/models/student"}
CLUSTER = {"ROLLOUT_NNODES": 1, "NODE_GPUS": 4}
DISTILL = {**STUDENT, **CLUSTER, "TRAINER_NNODES": 1, "TEACHER_NNODES": 1, "TEACHER": "/models/teacher"}
STEPS = {
    "eval.sh": {**STUDENT, **CLUSTER},
    "collect.sh": {"MODEL": "/models/teacher", **CLUSTER},
    "distill.sh": DISTILL,
}
GPUS = ["generic", "h100", "gb300"]


def key(arg: str) -> str:
    return arg.split("=", 1)[0].lstrip("+")


def value(args: list[str], name: str) -> str:
    (match,) = [arg.split("=", 1)[1] for arg in args if key(arg) == name]
    return match


def bash(snippet: str, **variables) -> str:
    env = {"PATH": "/usr/bin:/bin", "WORK_DIR": "/work", **variables}
    result = subprocess.run(
        ["bash", "-c", f"source {ROOT}/scripts/common.sh\n{snippet}"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_merge_args_keeps_the_last_value_of_each_key():
    out = bash("A=(a=1 +b=2 c.d=3 c.e=4 a=5 c={x:1} +b=6)\nmerge_args A\nprintf '%s\\n' \"${A[@]}\"")
    assert out.split() == ["a=5", "c={x:1}", "+b=6"]


@pytest.mark.parametrize("gpu", GPUS)
@pytest.mark.parametrize("step", STEPS)
def test_overrides_are_unique(run_step, step, gpu):
    module, args = run_step(step, GPU=gpu, **STEPS[step])
    assert module == f"opd.{step.removesuffix('.sh')}"
    assert value(args, "step") == step.removesuffix(".sh")  # the step's overlay
    keys = [key(arg) for arg in args]
    assert len(keys) == len(set(keys)), "Hydra refuses two values for one key"


@pytest.mark.parametrize("step", STEPS)
def test_generic_hardware_turns_off_pears_tuned_kernels(run_step, step):
    _, args = run_step(step, GPU="generic", **STEPS[step])
    assert value(args, "actor_rollout_ref.rollout.kernels") == "null"
    assert value(args, "actor_rollout_ref.actor.fsdp_config.use_gpu_backend") == "false"
    assert value(args, "actor_rollout_ref.model.override_config.attn_implementation") == "flash_attention_2"
    assert value(args, "actor_rollout_ref.rollout.custom.lmcache") == "{enable: false}"
    assert value(args, "actor_rollout_ref.rollout.n_gpus_per_node") == "4"
    assert value(args, "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size") == "4"


def test_h100_keeps_pears_recipe(run_step):
    _, args = run_step("eval.sh", GPU="h100", **STEPS["eval.sh"])
    assert "flash_attention_3" in value(args, "actor_rollout_ref.model.override_config.attn_implementation")
    assert "enable: true" in value(args, "actor_rollout_ref.rollout.custom.lmcache")
    assert value(args, "actor_rollout_ref.rollout.n_gpus_per_node") == "8"


def test_students_use_the_teachers_chat_template(run_step, tmp_path):
    _, student = run_step("eval.sh", **STEPS["eval.sh"])
    expected = tmp_path / "work" / "models" / "Qwen3.8-27B-PEAR-LIBERO_only" / "chat_template.jinja"
    assert value(student, "chat_template_file") == str(expected)
    _, teacher = run_step("eval.sh", **{**STEPS["eval.sh"], "MODEL_FILE": MODELS / "teacher.sh"})
    assert "chat_template_file" not in map(key, teacher)


def test_collect_runs_the_teacher_at_xhigh_unless_overridden(run_step):
    _, args = run_step("collect.sh", **STEPS["collect.sh"])
    assert value(args, "data.apply_chat_template_kwargs.reasoning_effort") == "xhigh"
    _, args = run_step("collect.sh", ["+data.apply_chat_template_kwargs.reasoning_effort=low"], **STEPS["collect.sh"])
    assert value(args, "data.apply_chat_template_kwargs.reasoning_effort") == "low"


def test_user_overrides_replace_the_hardware_profile(run_step):
    _, args = run_step("eval.sh", ["actor_rollout_ref.rollout.kernels=custom"], **STEPS["eval.sh"])
    assert value(args, "actor_rollout_ref.rollout.kernels") == "custom"


@pytest.mark.parametrize(
    ("bench", "episodes"),
    [("libero", "libero_all.json"), ("libero_pro", "all.json"), ("libero_plus", "all.json")],
)
def test_benchmarks_use_their_own_episodes_tools_and_simulator(run_step, bench, episodes, tmp_path):
    _, args = run_step("eval.sh", BENCH=bench, **STEPS["eval.sh"])
    configs = PEAR / "pear" / "envs" / bench / "configs"
    assert value(args, "data.train_files") == f"[{configs / episodes}]"
    assert value(args, "actor_rollout_ref.rollout.multi_turn.tool_config_path") == str(configs / "tool_config.yaml")
    env = "ray_kwargs.ray_init.runtime_env.env_vars"
    assert value(args, f"{env}.LIBERO_PYTHON") == str(tmp_path / "work" / "envs" / bench / "bin" / "python")
    assert (f"{env}.MAGICK_HOME" in map(key, args)) == (bench == "libero_plus")


@pytest.mark.parametrize("step", STEPS)
def test_every_referenced_repository_file_exists(run_step, step):
    _, args = run_step(step, **STEPS[step])
    paths = [
        part
        for arg in args
        for part in arg.split("=", 1)[1].strip("[]'").split(",")
        if part.startswith(str(PEAR)) and "sitecustomize" not in part
    ]
    assert paths
    missing = [path for path in paths if not Path(path).exists()]
    assert not missing


def test_distill_sizes_the_trainer_and_the_teacher(run_step):
    _, args = run_step("distill.sh", TRAINER_GPUS=2, TEACHER_GPUS=1, ROLLOUT_GPUS=1, **DISTILL)
    assert value(args, "trainer.n_gpus_per_node") == "2"
    assert value(args, "distillation.n_gpus_per_node") == "1"
    assert value(args, "actor_rollout_ref.rollout.n_gpus_per_node") == "1"
    assert value(args, "distillation.teacher_models.teacher_model.model_path") == "/models/teacher"
    assert value(args, "trainer.total_training_steps") == "200"
    assert "actor_rollout_ref.actor.ppo_max_token_len_per_gpu" in map(key, args)  # the model's training arguments


def test_sft(run_step, tmp_path):
    module, args = run_step(
        "sft.sh", ["trainer.total_epochs=1"], MODEL="/models/student", EPISODES_DIR=tmp_path / "ep", NGPUS=2
    )
    assert module == "opd.sft"
    assert value(args, "data.train_files") == str(tmp_path / "ep")
    assert value(args, "model.path") == "/models/student"
    assert value(args, "trainer.total_epochs") == "1"


@pytest.mark.parametrize("missing", ["MODEL", "RUN", "ROLLOUT_NNODES"])
def test_a_missing_setting_stops_the_step(verl_env, tmp_path, missing):
    env = {"PATH": "/usr/bin:/bin", "WORK_DIR": str(tmp_path), "VERL_ENV": str(verl_env), "RUN": str(tmp_path / "r")}
    env.update({k: str(v) for k, v in STEPS["eval.sh"].items()})
    env.pop(missing)
    result = subprocess.run(["bash", str(ROOT / "scripts" / "eval.sh")], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert missing in result.stderr


@pytest.mark.parametrize("model_file", sorted(MODELS.glob("*.sh")), ids=lambda path: path.stem)
def test_model_files_name_one_pinned_revision(model_file):
    # PEAR's download script reads these two assignments with sed: each must appear once, the revision a commit
    text = model_file.read_text().splitlines()
    repos = [line.split("=", 1)[1] for line in text if line.startswith("MODEL_REPO=")]
    revisions = [line.split("=", 1)[1] for line in text if line.startswith("MODEL_REVISION=")]
    assert len(repos) == 1 and len(revisions) == 1
    assert len(revisions[0]) == 40 and int(revisions[0], 16) >= 0


@pytest.mark.network
@pytest.mark.parametrize("model_file", sorted(MODELS.glob("*.sh")), ids=lambda path: path.stem)
def test_pinned_revisions_exist_on_the_hub(model_file):
    lines = model_file.read_text().splitlines()
    repo = next(line.split("=", 1)[1] for line in lines if line.startswith("MODEL_REPO="))
    revision = next(line.split("=", 1)[1] for line in lines if line.startswith("MODEL_REVISION="))
    request = urllib.request.Request(f"https://huggingface.co/{repo}/resolve/{revision}/config.json", method="HEAD")
    assert urllib.request.urlopen(request, timeout=60).status == 200


def test_pear_is_pinned():
    # the submodule's commit is the one the configs and the recipes were written against
    result = subprocess.run(["git", "-C", str(PEAR), "rev-parse", "HEAD"], capture_output=True, text=True)
    assert result.stdout.strip() == "faff47472fe2a4f66ed6503e9a711fa239ced223"

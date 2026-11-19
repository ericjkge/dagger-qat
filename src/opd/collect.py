"""Step 1: the teacher's evaluation, which also writes each successful episode as an SFT example (opd.sft_data): the
exact tensors the rollout fed the model."""

from pathlib import Path

import hydra
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from pear.eval.embodied import EmbodiedEvaluationTrainer
from pear.trainer.config import RunConfig
from pear.trainer.runner import PearTaskRunner
from verl.trainer.main_ppo import run_ppo

from opd import PEAR_CONFIGS, use_chat_template_file
from opd.sft_data import sft_example

# what verl's agent loop stores per trajectory (verl/trainer/ppo/v1/agent_loop_tq.py); loss_mask is the response mask
FIELDS = ["input_ids", "position_ids", "loss_mask", "multi_modal_inputs"]


class CollectionTrainer(EmbodiedEvaluationTrainer):
    """PEAR's evaluation, which also writes <collect.out_dir>/<uid>.pt for every successful episode."""

    def _records(self, batch: tq.KVBatchMeta) -> list[dict]:
        records = super()._records(batch)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=FIELDS)
        out_dir = Path(self.config.collect.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # tensor columns are stacked or jagged, unbinding to each row's own length; the non-tensor column comes back
        # as a list-like of the backend's type
        rows = zip(
            records,
            data["input_ids"].unbind(),
            data["position_ids"].unbind(),
            data["loss_mask"].unbind(),
            list(data["multi_modal_inputs"]),
            strict=True,
        )
        for record, input_ids, position_ids, response_mask, multi_modal_inputs in rows:
            if record["score"] > 0:
                example = sft_example(input_ids, position_ids, response_mask, multi_modal_inputs)
                torch.save(example, out_dir / f"{record['uid']}.pt")

        return records


@ray.remote
class CollectionRunner(PearTaskRunner):
    def build_trainer(self, config: DictConfig) -> CollectionTrainer:
        return CollectionTrainer(config=config)

    def teacher_client(self, trainer: CollectionTrainer) -> None:
        return None


@hydra.main(config_path=PEAR_CONFIGS, config_name="embodied", version_base=None)
def main(config: DictConfig) -> None:
    use_chat_template_file(config)
    RunConfig(config).prepare()
    run_ppo(config, task_runner_class=CollectionRunner)


if __name__ == "__main__":
    main()

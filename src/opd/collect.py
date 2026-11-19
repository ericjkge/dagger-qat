"""Step 1: the teacher's evaluation, also saving each successful episode as the exact token sequence the rollout
produced (prompt and response ids, the mask of the tokens the teacher generated) with its images, for SFT."""

from pathlib import Path

import hydra
import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from pear.eval.embodied import EmbodiedEvaluationTrainer
from pear.trainer.config import RunConfig
from pear.trainer.runner import PearTaskRunner
from verl.trainer.main_ppo import run_ppo

from opd import CONFIGS, use_chat_template_file

FIELDS = ["prompts", "responses", "response_mask", "multi_modal_data"]


class CollectionTrainer(EmbodiedEvaluationTrainer):
    """PEAR's evaluation, which also writes <out_dir>/<uid>.pt for every successful episode."""

    def _records(self, batch: tq.KVBatchMeta) -> list[dict]:
        records = super()._records(batch)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=FIELDS)
        out_dir = Path(self.config.collect.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # jagged columns unbind to their rows' own lengths
        rows = zip(
            records,
            data["prompts"].unbind(),
            data["responses"].unbind(),
            data["response_mask"].unbind(),
            data["multi_modal_data"],
            strict=True,
        )
        for record, prompt, response, response_mask, multi_modal in rows:
            if record["score"] > 0:
                episode = {
                    "prompt_ids": prompt.clone(),
                    "response_ids": response.clone(),
                    "response_mask": response_mask.clone(),
                    "images": [np.asarray(image) for image in multi_modal.get("images", [])],
                    "data_source": record["data_source"],
                    "extra_fields": record["extra_fields"],
                }
                torch.save(episode, out_dir / f"{record['uid']}.pt")

        return records


@ray.remote
class CollectionRunner(PearTaskRunner):
    def build_trainer(self, config: DictConfig) -> CollectionTrainer:
        return CollectionTrainer(config=config)

    def teacher_client(self, trainer: CollectionTrainer) -> None:
        return None


@hydra.main(config_path=CONFIGS, config_name="collect", version_base=None)
def main(config: DictConfig) -> None:
    use_chat_template_file(config)
    RunConfig(config).prepare()
    run_ppo(config, task_runner_class=CollectionRunner)


if __name__ == "__main__":
    main()

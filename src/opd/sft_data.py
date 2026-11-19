"""The SFT examples of the teacher's successful episodes, one file each, as opd.collect writes them: the exact tensors
the rollout fed the model, with the loss on the tokens the teacher generated and none on the prompt or on what the
environment returned."""

from pathlib import Path

import torch
from torch.utils.data import Dataset


def sft_example(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    response_mask: torch.Tensor,
    multi_modal_inputs: dict[str, torch.Tensor],
) -> dict:
    """One rollout trajectory as an SFT example: input_ids (prompt and response), the rollout's (multimodal rotary)
    position ids, and a loss mask over the whole sequence that is the response mask on the response, 0 on the prompt."""
    prompt_length = len(input_ids) - len(response_mask)
    if prompt_length < 0 or position_ids.shape[-1] != len(input_ids):
        raise ValueError(
            f"inconsistent trajectory: {len(input_ids)} ids, {len(response_mask)} response mask entries, "
            f"position ids of shape {tuple(position_ids.shape)}"
        )
    loss_mask = torch.cat([torch.zeros(prompt_length, dtype=response_mask.dtype), response_mask])
    return {
        "input_ids": input_ids.clone(),
        "position_ids": position_ids.clone(),
        "loss_mask": loss_mask,
        "multi_modal_inputs": {key: value.clone() for key, value in multi_modal_inputs.items()},
    }


class EpisodeDataset(Dataset):
    """verl's SFT dataset (data.custom_cls, pad_mode no_padding) over a directory of SFT examples."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        # verl's SFT trainer passes data.train_files as parquet_files: here, the directory opd.collect wrote
        directory = Path(parquet_files[0] if isinstance(parquet_files, list | tuple) else parquet_files)
        self.paths = sorted(directory.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no episodes (*.pt) in {directory}")
        if max_samples > 0:
            self.paths = self.paths[:max_samples]
        self.max_length = config.max_length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        example = torch.load(self.paths[index], weights_only=True)
        if len(example["input_ids"]) > self.max_length:
            length = len(example["input_ids"])
            raise ValueError(f"{self.paths[index]}: {length} tokens > data.max_length={self.max_length}")
        return example

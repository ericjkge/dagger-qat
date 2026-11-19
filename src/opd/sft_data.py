"""The SFT dataset of the teacher's episodes (opd.collect): each item is the episode's exact token sequence, the loss on
the tokens the teacher generated and none on the prompt or the environment's observations."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from verl.models.transformers.qwen2_vl import get_rope_index


def episode_tensors(episode: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The input ids of an episode and its loss mask: 1 on every response token the model generated."""
    input_ids = torch.cat([episode["prompt_ids"], episode["response_ids"]])
    loss_mask = torch.cat([torch.zeros_like(episode["prompt_ids"]), episode["response_mask"]])
    return input_ids, loss_mask


class EpisodeDataset(Dataset):
    """verl's SFT dataset contract (data.custom_cls) in no-padding mode, over a directory of episode files."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples: int = -1):
        directory = parquet_files[0] if isinstance(parquet_files, list | tuple) else parquet_files
        self.paths = sorted(Path(directory).glob("*.pt"))
        if max_samples > 0:
            self.paths = self.paths[:max_samples]
        self.processor = processor
        self.max_length = config.max_length

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        episode = torch.load(self.paths[index], weights_only=False)
        input_ids, loss_mask = episode_tensors(episode)
        if len(input_ids) > self.max_length:
            raise ValueError(f"{self.paths[index]}: {len(input_ids)} tokens > data.max_length={self.max_length}")

        images = [Image.fromarray(image) for image in episode["images"]]
        multi_modal_inputs = dict(self.processor.image_processor(images=images, return_tensors="pt"))
        # Qwen-VL's multimodal rotary positions: the text positions over the three vision ones, as verl's own dataset
        vision_position_ids = get_rope_index(
            self.processor,
            input_ids=input_ids,
            image_grid_thw=multi_modal_inputs["image_grid_thw"],
            attention_mask=torch.ones_like(input_ids),
        )
        text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)

        return {
            "input_ids": input_ids,
            "position_ids": torch.cat([text_position_ids, vision_position_ids]),
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
        }

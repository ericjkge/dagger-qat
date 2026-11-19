"""The SFT examples opd.collect writes and the dataset verl's SFT trainer reads them through."""

import pytest
import torch
from omegaconf import OmegaConf

from opd.sft_data import EpisodeDataset, sft_example


def trajectory(prompt: int, response_mask: list[int]):
    length = prompt + len(response_mask)
    return (
        torch.arange(100, 100 + length),
        torch.arange(length).repeat(4, 1),  # (text + 3 vision) rotary positions
        torch.tensor(response_mask),
        {"pixel_values": torch.ones(8, 3), "image_grid_thw": torch.tensor([[1, 2, 4]])},
    )


def test_loss_falls_on_generated_tokens_only():
    # 3 prompt tokens; the response interleaves generated tokens (1) and an observation (0)
    example = sft_example(*trajectory(3, [1, 1, 0, 0, 1]))
    assert example["loss_mask"].tolist() == [0, 0, 0, 1, 1, 0, 0, 1]
    assert len(example["input_ids"]) == len(example["loss_mask"]) == example["position_ids"].shape[-1]
    assert example["multi_modal_inputs"]["image_grid_thw"].tolist() == [[1, 2, 4]]


def test_example_owns_its_storage():
    # a row unbound from a batch is a view: saved as is, it would carry the whole batch's storage
    batch = torch.arange(20).view(2, 10)
    input_ids, position_ids, response_mask, multi_modal_inputs = trajectory(4, [1] * 6)
    example = sft_example(batch[0], position_ids, response_mask, multi_modal_inputs)
    assert example["input_ids"].untyped_storage().nbytes() == 10 * batch.element_size()


@pytest.mark.parametrize(
    ("ids", "positions", "mask"),
    [(5, 5, 6), (5, 4, 3)],  # a response longer than the sequence; positions of another length
)
def test_inconsistent_trajectories_are_refused(ids, positions, mask):
    with pytest.raises(ValueError):
        sft_example(torch.arange(ids), torch.zeros(4, positions), torch.ones(mask, dtype=torch.long), {})


def test_dataset_reads_what_collect_writes(tmp_path):
    for index in range(3):
        torch.save(sft_example(*trajectory(3 + index, [1, 0, 1])), tmp_path / f"{index}.pt")
    config = OmegaConf.create({"max_length": 100})

    dataset = EpisodeDataset([str(tmp_path)], tokenizer=None, config=config, max_samples=2)
    assert len(dataset) == 2
    item = dataset[1]
    assert set(item) == {"input_ids", "position_ids", "loss_mask", "multi_modal_inputs"}
    assert item["loss_mask"].tolist() == [0, 0, 0, 0, 1, 0, 1]

    with pytest.raises(ValueError, match="max_length"):
        EpisodeDataset(str(tmp_path), None, OmegaConf.create({"max_length": 5}))[2]


def test_dataset_refuses_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        EpisodeDataset(str(tmp_path), None, OmegaConf.create({"max_length": 10}))

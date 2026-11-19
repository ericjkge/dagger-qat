import pytest
import torch

sft_data = pytest.importorskip("opd.sft_data")  # needs PEAR's verl environment


def test_loss_only_on_generated_tokens():
    episode = {
        "prompt_ids": torch.tensor([1, 2, 3]),
        "response_ids": torch.tensor([4, 5, 6, 7]),
        "response_mask": torch.tensor([1, 1, 0, 1]),  # 6 is an observation the environment returned
    }
    input_ids, loss_mask = sft_data.episode_tensors(episode)
    assert input_ids.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert loss_mask.tolist() == [0, 0, 0, 1, 1, 0, 1]

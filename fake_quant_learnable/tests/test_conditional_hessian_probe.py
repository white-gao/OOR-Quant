import torch

from fake_quant_learnable.probe_conditional_hessian import (
    channelwise_cosine,
    conditional_mixture,
    normalized_entropy,
)


def test_conditional_mixture_detects_group_specialized_channels() -> None:
    # rows are slot groups; columns are Linear output channels
    energy = torch.tensor(
        [
            [10.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    pi = conditional_mixture(energy)

    torch.testing.assert_close(pi[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(pi[1], torch.full((5,), 0.2))
    entropy = normalized_entropy(pi)
    assert entropy[0].item() < 1e-5
    assert entropy[1].item() > 0.999


def test_channelwise_cosine_is_one_for_identical_slot_mixtures() -> None:
    pi = torch.tensor([[0.7, 0.1, 0.1, 0.05, 0.05], [0.2, 0.2, 0.2, 0.2, 0.2]])
    torch.testing.assert_close(channelwise_cosine(pi, pi), torch.ones(2))

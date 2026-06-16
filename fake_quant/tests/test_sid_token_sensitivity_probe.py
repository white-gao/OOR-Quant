from __future__ import annotations

import torch

from fake_quant.probes.token_sensitivity import plot_prefill_sid_sensitivity as token_plot
from fake_quant.probes.token_sensitivity.plot_prefill_sid_sensitivity import (
    compute_token_sensitivity,
    label_prefill_tokens,
)
from fake_quant.probes.token_sensitivity.plot_prefill_sid_sensitivity_2d import (
    compute_channel_profiles,
    compute_token_channel_sensitivity,
)
from fake_quant.probes.token_sensitivity.probe_sid_token_sensitivity import (
    SensitivityAccumulator,
    label_teacher_forced_tokens,
)


def test_label_teacher_forced_tokens_marks_history_and_prediction_positions() -> None:
    token_texts = [
        "<|im_start|>",
        "user",
        "Recommend",
        "<|sid_begin|>",
        "<s_a_1>",
        "<s_b_2>",
        "<s_c_3>",
        "<|sid_end|>",
        "next",
        "<|sid_begin|>",
        "<s_a_9>",
        "<s_b_8>",
    ]

    groups = label_teacher_forced_tokens(token_texts, prompt_len=10, target_len=3)

    assert groups[2] == "text_prompt"
    assert groups[3] == "history_sid_boundary"
    assert groups[4] == "history_sid_a"
    assert groups[5] == "history_sid_b"
    assert groups[6] == "history_sid_c"
    assert groups[7] == "history_sid_boundary"
    assert groups[9] == "predict_s_a_position"
    assert groups[10] == "predict_s_b_position"
    assert groups[11] == "predict_s_c_position"


def test_sensitivity_accumulator_computes_group_means() -> None:
    acc = SensitivityAccumulator()
    activation = torch.tensor(
        [[[1.0, -2.0], [3.0, 4.0], [5.0, -6.0]]],
        dtype=torch.float32,
    )
    grad = torch.tensor(
        [[[0.5, -1.0], [2.0, -3.0], [4.0, 5.0]]],
        dtype=torch.float32,
    )
    groups = ["text_prompt", "predict_s_a_position", None]

    acc.add("layer27.block_output", activation, grad, groups)
    rows = acc.rows()

    by_group = {row["token_group"]: row for row in rows}
    assert by_group["text_prompt"]["num_tokens"] == 1
    assert by_group["text_prompt"]["mean_abs_grad"] == 0.75
    assert by_group["text_prompt"]["mean_abs_act_grad"] == 1.25
    assert by_group["predict_s_a_position"]["num_tokens"] == 1
    assert by_group["predict_s_a_position"]["mean_abs_grad"] == 2.5
    assert by_group["predict_s_a_position"]["mean_abs_act_grad"] == 9.0


def test_label_prefill_tokens_marks_final_sid_begin_as_predict_s_a() -> None:
    token_texts = [
        "<|im_start|>",
        "user",
        "history",
        "<|sid_begin|>",
        "<s_a_1>",
        "<s_b_2>",
        "<s_c_3>",
        "<|sid_end|>",
        "next",
        "<|sid_begin|>",
    ]

    groups = label_prefill_tokens(token_texts)

    assert groups[2] == "text_prompt"
    assert groups[3] == "history_sid_boundary"
    assert groups[4] == "history_sid_a"
    assert groups[5] == "history_sid_b"
    assert groups[6] == "history_sid_c"
    assert groups[7] == "history_sid_boundary"
    assert groups[9] == "predict_s_a_position"


def test_compute_token_sensitivity_returns_mean_abs_act_grad_per_token() -> None:
    activation = torch.tensor(
        [[[1.0, -2.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    grad = torch.tensor(
        [[[0.5, -1.0], [2.0, -3.0]]],
        dtype=torch.float32,
    )

    sensitivity = compute_token_sensitivity(activation, grad)

    torch.testing.assert_close(sensitivity, torch.tensor([1.25, 9.0]))


def test_compute_token_channel_sensitivity_keeps_full_matrix() -> None:
    activation = torch.tensor(
        [[[1.0, -2.0], [3.0, 4.0]]],
        dtype=torch.float32,
    )
    grad = torch.tensor(
        [[[0.5, -1.0], [2.0, -3.0]]],
        dtype=torch.float32,
    )

    matrix = compute_token_channel_sensitivity(activation, grad)

    assert tuple(matrix.shape) == (2, 2)
    torch.testing.assert_close(matrix, torch.tensor([[0.5, 2.0], [6.0, 12.0]]))


def test_compute_channel_profiles_groups_tokens_without_dropping_channels() -> None:
    matrix = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 20.0, 30.0]],
        dtype=torch.float32,
    )
    groups = ["text_prompt", "text_prompt", "predict_s_a_position"]

    profiles = compute_channel_profiles(matrix, groups)

    torch.testing.assert_close(profiles["text_prompt"], torch.tensor([2.5, 3.5, 4.5]))
    torch.testing.assert_close(profiles["predict_s_a_position"], torch.tensor([10.0, 20.0, 30.0]))



def test_prefill_plot_exposes_only_s_a_loss_path() -> None:
    assert not hasattr(token_plot, "build_sid_loss_example")
    assert not hasattr(token_plot, "label_loss_tokens")
    assert "loss_mode" not in token_plot.parse_args.__code__.co_names

from __future__ import annotations

from types import SimpleNamespace

import torch

from fake_quant.run_ad_sid import PROJECT_ROOT, result_path
from fake_quant.run_ad_sid import generate_batch


def test_result_path_resolves_relative_output_dir_from_project_root(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)

    output = result_path(
        "fake_quant/results/v1.0/results_demo",
        "OneRec-demo",
        "test",
    )

    assert output.is_absolute()
    assert output == (
        PROJECT_ROOT
        / "fake_quant/results/v1.0/results_demo/OneRec-demo/ad/test_generated.json"
    )


class _BatchTokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def __init__(self) -> None:
        self.padding_side = "right"
        self.calls = []

    def __call__(self, prompts, *, return_tensors: str, padding: bool = False):
        self.calls.append((list(prompts), return_tensors, padding, self.padding_side))
        encoded = {
            "short": [1],
            "long": [2, 3],
        }
        ids = [encoded[prompt] for prompt in prompts]
        width = max(len(item) for item in ids)
        padded = []
        masks = []
        for item in ids:
            pad_len = width - len(item)
            if self.padding_side == "left":
                padded.append([self.pad_token_id] * pad_len + item)
                masks.append([0] * pad_len + [1] * len(item))
            else:
                padded.append(item + [self.pad_token_id] * pad_len)
                masks.append([1] * len(item) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(padded),
            "attention_mask": torch.tensor(masks),
        }

    def decode(self, ids, *, skip_special_tokens: bool = False) -> str:
        return ",".join(str(int(item)) for item in ids.tolist())


class _BatchGenerateModel:
    def __init__(self) -> None:
        self.last_input_ids = None
        self.last_attention_mask = None

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        self.last_input_ids = input_ids.detach().cpu()
        self.last_attention_mask = kwargs["attention_mask"].detach().cpu()
        num_return_sequences = kwargs["num_return_sequences"]
        rows = []
        for batch_idx in range(input_ids.shape[0]):
            for return_idx in range(num_return_sequences):
                generated = torch.tensor(
                    [10 * batch_idx + return_idx + 1],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                rows.append(torch.cat([input_ids[batch_idx], generated]))
        return torch.stack(rows)


def test_generate_batch_left_pads_and_groups_return_sequences() -> None:
    tokenizer = _BatchTokenizer()
    model = _BatchGenerateModel()
    args = SimpleNamespace(max_new_tokens=1, num_beams=2, num_return_sequences=2)

    generations = generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=["short", "long"],
        input_device=torch.device("cpu"),
        args=args,
    )

    assert generations == [["1", "2"], ["11", "12"]]
    assert tokenizer.calls == [(["short", "long"], "pt", True, "left")]
    assert tokenizer.padding_side == "right"
    torch.testing.assert_close(model.last_input_ids, torch.tensor([[0, 1], [2, 3]]))
    torch.testing.assert_close(model.last_attention_mask, torch.tensor([[0, 1], [1, 1]]))

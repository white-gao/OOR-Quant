from __future__ import annotations


def test_package_exports_only_active_ptq_symbols() -> None:
    import fake_quant_learnable as fql

    forbidden_fragments = ("Learnable", "learnable", "lwt", "let", "tail", "calibrate")

    assert "BaselineFakeQuantLinear" in fql.__all__
    assert "SmoothQuantFakeQuantLinear" in fql.__all__
    assert "GPTQFakeQuantLinear" in fql.__all__
    assert "apply_baseline_w8a8" in fql.__all__
    assert "collect_smoothquant_scales" in fql.__all__
    assert "gptq_fp8_quantize_weight" in fql.__all__
    assert "PromptTokenWeightConfig" in fql.__all__
    assert "build_prompt_token_weight_batches" in fql.__all__
    assert "GradientTokenWeightConfig" in fql.__all__
    assert "collect_gradient_token_weight_batches_by_layer" in fql.__all__
    assert not any(
        any(fragment in exported for fragment in forbidden_fragments)
        for exported in fql.__all__
    )


def test_runner_modes_are_limited_to_active_ptq_methods(monkeypatch) -> None:
    from fake_quant_learnable.run_m1_onerec_ad import parse_args

    monkeypatch.setattr("sys.argv", ["prog"])
    args = parse_args()

    assert args.mode == "baseline_w8a8"
    assert not hasattr(args, "lwt_lr")
    assert not hasattr(args, "let_lr")
    assert not hasattr(args, "epochs")

    monkeypatch.setattr("sys.argv", ["prog", "--mode", "gptq_fp8_w8a8"])
    args = parse_args()
    assert args.mode == "gptq_fp8_w8a8"

    monkeypatch.setattr("sys.argv", ["prog", "--mode", "weighted_gptq_fp8_w8a8"])
    args = parse_args()
    assert args.mode == "weighted_gptq_fp8_w8a8"

    monkeypatch.setattr("sys.argv", ["prog", "--mode", "weighted_gptq_fp8_w8a8_tail1"])
    args = parse_args()
    assert args.mode == "weighted_gptq_fp8_w8a8_tail1"

    monkeypatch.setattr("sys.argv", ["prog", "--mode", "grad_weighted_gptq_fp8_w8a8"])
    args = parse_args()
    assert args.mode == "grad_weighted_gptq_fp8_w8a8"

    monkeypatch.setattr("sys.argv", ["prog", "--mode", "m2_lwt_let"])
    try:
        parse_args()
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("m2_lwt_let should not be an active runner mode")



def test_auxiliary_code_lives_under_support_package() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]

    assert not (root / "smoothquant_runtime.py").exists()
    assert not (root / "runtime_utils.py").exists()
    assert not (root / "inspect_smoothquant_distribution.py").exists()
    assert (root / "support" / "smoothquant_runtime.py").exists()
    assert (root / "support" / "runtime_utils.py").exists()
    assert (root / "support" / "inspect_smoothquant_distribution.py").exists()

    from fake_quant_learnable.support.smoothquant_runtime import collect_smoothquant_scales

    assert callable(collect_smoothquant_scales)

from __future__ import annotations

from fake_quant.run_ad_sid import PROJECT_ROOT, result_path


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

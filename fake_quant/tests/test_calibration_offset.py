from __future__ import annotations

from fake_quant.smoothquant.collect_smooth_scales import slice_calibration_data


def test_slice_calibration_data_uses_offset_and_size() -> None:
    data = {str(i): {"metadata": {"row_index": i}} for i in range(1300)}

    calib128 = slice_calibration_data(data, sample_size=128, sample_offset=1000)
    calib256 = slice_calibration_data(data, sample_size=256, sample_offset=1000)

    assert list(calib128) == [str(i) for i in range(1000, 1128)]
    assert list(calib256) == [str(i) for i in range(1000, 1256)]
    assert set(calib128).isdisjoint({str(i) for i in range(1000)})
    assert set(calib256).isdisjoint({str(i) for i in range(1000)})


def test_slice_calibration_data_rejects_short_dataset() -> None:
    data = {str(i): {} for i in range(1100)}

    try:
        slice_calibration_data(data, sample_size=128, sample_offset=1000)
    except ValueError as exc:
        assert "Not enough samples" in str(exc)
    else:
        raise AssertionError("Expected ValueError for insufficient calibration data.")

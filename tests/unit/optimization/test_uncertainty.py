import numpy as np
import pandas as pd
import pytest

from demand_engine.optimization.uncertainty import protection_error_stats


def _preds(errors: list[float], uid: str = "a", cutoff: str = "2020-01-01") -> pd.DataFrame:
    n = len(errors)
    return pd.DataFrame(
        {
            "unique_id": uid,
            "cutoff": pd.Timestamp(cutoff),
            "ds": pd.date_range(pd.Timestamp(cutoff) + pd.Timedelta(days=1), periods=n),
            "y": np.array(errors) + 5.0,
            "yhat": 5.0,
        }
    )


def test_constant_under_forecast_is_pure_bias() -> None:
    stats = protection_error_stats(_preds([1.0] * 20), days=4)
    assert stats.loc["a", "bias"] == pytest.approx(4.0)  # every 4-day error sum is +4
    assert stats.loc["a", "sigma"] == pytest.approx(0.0)


def test_alternating_errors_cancel_over_even_windows() -> None:
    stats = protection_error_stats(_preds([1.0, -1.0] * 10), days=2)
    assert stats.loc["a", "bias"] == pytest.approx(0.0)
    assert stats.loc["a", "sigma"] == pytest.approx(0.0)


def test_windows_do_not_cross_folds() -> None:
    folds = [_preds([1.0] * 3, cutoff=f"2020-{m:02d}-01") for m in range(1, 4)]
    folds += [_preds([-1.0] * 3, cutoff=f"2020-{m:02d}-01") for m in range(4, 7)]
    stats = protection_error_stats(pd.concat(folds), days=3)
    # each fold contributes exactly one window: three of +3 and three of -3
    assert stats.loc["a", "bias"] == pytest.approx(0.0)
    assert stats.loc["a", "sigma"] == pytest.approx(3.0)


def test_fallback_to_iid_scaling_for_short_series() -> None:
    stats = protection_error_stats(_preds([2.0, -2.0]), days=4)
    assert stats.loc["a", "bias"] == pytest.approx(0.0)
    assert stats.loc["a", "sigma"] == pytest.approx(np.sqrt(4) * 2.0)


def test_invalid_days() -> None:
    with pytest.raises(ValueError, match=">= 1"):
        protection_error_stats(_preds([1.0]), days=0)

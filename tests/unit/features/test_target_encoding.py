import numpy as np
import pandas as pd
import pytest

from demand_engine.features.target_encoding import TE_FEATURES, add_target_encodings


def _frame() -> pd.DataFrame:
    """Two stores selling item A; item B in store S1 only. 6 days."""
    ds = pd.date_range("2020-01-01", periods=6)
    rows = []
    for store, item, ys in [("S1", "A", [1, 2, 3, 4, 5, 6]), ("S2", "A", [10, 10, 10, 10, 10, 10]),
                            ("S1", "B", [0, 0, 6, 0, 0, 6])]:  # fmt: skip
        for d, y in zip(ds, ys, strict=True):
            rows.append({"unique_id": f"{item}_{store}", "item_id": item, "store_id": store,
                         "dept_id": "D", "wday": d.dayofweek, "ds": d, "y": float(y)})  # fmt: skip
    return pd.DataFrame(rows)


def test_values_hand_computed() -> None:
    out = add_target_encodings(_frame(), horizon=2)
    a1 = out[(out["unique_id"] == "A_S1")].set_index("ds")["te_item"]
    # day 3 (2020-01-03) may use days <= 2020-01-01: item A = (1 + 10) / 2
    assert a1[pd.Timestamp("2020-01-03")] == pytest.approx(5.5)
    # day 6 uses days <= 2020-01-04: (1+2+3+4 + 4*10) / 8
    assert a1[pd.Timestamp("2020-01-06")] == pytest.approx(50 / 8)
    # store-dept S1/D on day 5 uses days <= 2020-01-03: A_S1 (1,2,3) + B_S1 (0,0,6) -> 12/6
    s1 = out[out["unique_id"] == "A_S1"].set_index("ds")["te_store_dept"]
    assert s1[pd.Timestamp("2020-01-05")] == pytest.approx(2.0)


def test_no_history_old_enough_gives_nan() -> None:
    out = add_target_encodings(_frame(), horizon=2)
    early = out[out["ds"] < pd.Timestamp("2020-01-03")]
    assert early[list(TE_FEATURES)].isna().all().all()


def test_never_uses_data_after_t_minus_h() -> None:
    """Changing y on day s must not change any encoding dated before s + h."""
    df = _frame()
    h = 2
    base = add_target_encodings(df, h)
    for s in df["ds"].unique():
        changed = df.copy()
        changed.loc[changed["ds"] == s, "y"] += 1000.0
        out = add_target_encodings(changed, h)
        safe = df["ds"] < s + pd.Timedelta(days=h)
        pd.testing.assert_frame_equal(
            out.loc[safe, list(TE_FEATURES)], base.loc[safe, list(TE_FEATURES)]
        )


def test_future_nan_targets_are_ignored() -> None:
    df = _frame()
    df.loc[df["ds"] > pd.Timestamp("2020-01-04"), "y"] = np.nan
    out = add_target_encodings(df, horizon=2)
    assert out.loc[out["ds"] == pd.Timestamp("2020-01-06"), "te_item"].notna().all()


def test_validation() -> None:
    with pytest.raises(ValueError, match="horizon"):
        add_target_encodings(_frame(), horizon=0)
    with pytest.raises(ValueError, match="needs columns"):
        add_target_encodings(_frame().drop(columns="dept_id"), horizon=1)

import pandas as pd

from demand_engine.data.subset import allocate_proportionally, select_items
from demand_engine.data.synthetic import SyntheticM5


def test_allocation_sums_and_caps() -> None:
    alloc = allocate_proportionally({"A": 10, "B": 5, "C": 1}, 8)
    assert sum(alloc.values()) == 8
    assert all(alloc[k] <= v for k, v in {"A": 10, "B": 5, "C": 1}.items())


def test_allocation_more_than_available() -> None:
    assert allocate_proportionally({"A": 2, "B": 1}, 100) == {"A": 2, "B": 1}


def test_allocation_zero() -> None:
    assert allocate_proportionally({"A": 2}, 0) == {"A": 0}


def test_select_items_deterministic_and_stratified(synthetic_raw: SyntheticM5) -> None:
    a = select_items(synthetic_raw.sales, 6, seed=1)
    b = select_items(synthetic_raw.sales, 6, seed=1)
    assert a == b
    assert len(a) == len(set(a)) == 6
    cats = synthetic_raw.sales.drop_duplicates("item_id").set_index("item_id")["cat_id"]
    assert set(cats.loc[a]) <= set(cats)


def test_select_items_seed_changes_selection(synthetic_raw: SyntheticM5) -> None:
    picks = {tuple(select_items(synthetic_raw.sales, 4, seed=s)) for s in range(5)}
    assert len(picks) > 1


def test_select_items_caps_at_available() -> None:
    sales = pd.DataFrame({"item_id": ["x", "y", "x"], "cat_id": ["A", "A", "A"]})
    assert select_items(sales, 10, seed=0) == ["x", "y"]

"""Reproducible, category-stratified item sampling."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def allocate_proportionally(counts: dict[str, int], total: int) -> dict[str, int]:
    """Split ``total`` across groups in proportion to ``counts`` (largest-remainder method).

    Each group receives at most its own count, and the allocations sum to
    ``min(total, sum(counts))``.

    Args:
        counts: Available items per group.
        total: Number of items to allocate.

    Returns:
        Allocation per group.
    """
    available = sum(counts.values())
    target = min(total, available)
    if target == 0:
        return dict.fromkeys(counts, 0)
    keys = sorted(counts)
    exact = np.array([counts[k] * target / available for k in keys])
    alloc = np.floor(exact).astype(int)
    remainder = target - int(alloc.sum())
    order = np.argsort(-(exact - alloc), kind="stable")
    for idx in order[:remainder]:
        alloc[idx] += 1
    return {k: int(min(a, counts[k])) for k, a in zip(keys, alloc, strict=True)}


def select_items(sales: pd.DataFrame, n_items: int, seed: int) -> list[str]:
    """Pick ``n_items`` item ids stratified by ``cat_id``.

    Args:
        sales: Wide sales table with ``item_id`` and ``cat_id`` columns.
        n_items: Number of items to select (capped at the number available).
        seed: Random seed controlling which items are drawn.

    Returns:
        Sorted list of selected item ids.
    """
    items = sales[["item_id", "cat_id"]].drop_duplicates().sort_values("item_id")
    counts = items.groupby("cat_id").size().to_dict()
    allocation = allocate_proportionally({str(k): int(v) for k, v in counts.items()}, n_items)
    rng = np.random.default_rng(seed)
    chosen: list[str] = []
    for cat in sorted(allocation):
        pool = items.loc[items["cat_id"] == cat, "item_id"].to_numpy()
        chosen.extend(str(x) for x in rng.choice(pool, size=allocation[cat], replace=False))
    logger.info("Selected %d items: %s", len(chosen), allocation)
    return sorted(chosen)

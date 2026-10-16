# Defence demo script (≈ 8 minutes)

## Before the panel arrives (5 min, offline-safe)

```bash
uv run demand-engine demo          # embedded PostgreSQL + API + dashboard on :8000
```

* Open `http://127.0.0.1:8000` (dashboard) and `http://127.0.0.1:8000/docs` (API) in two tabs.
* Open `notebooks/02_results.ipynb` (already executed — no kernel needed to show it).
* Open a terminal in the repo for the live test run.
* **Fallback:** the pipeline results are stored as files in `artifacts/runs/<id>`; the demo
  needs no internet, no Docker and no re-training. If the browser fails, the notebook
  shows every number and chart.

## 1. The problem — 30 s

"Retailers must decide every day *how much stock to hold* for thousands of products.
Too little loses sales; too much ties up cash. This project forecasts demand for ~2,000
Walmart SKU-store pairs and turns forecasts into ordering decisions — and proves the
value by simulating 12 weeks of demand the models never saw."

## 2. Overview page — 1 min

* **76.8 % cost saving, fill rate 84 % → 98.6 %** versus a min/max rule of thumb.
* Point at the stacked bars: the rule of thumb loses money on **stock-outs** and on
  **too many small orders**; the optimised policies hold more stock but far fewer orders.
* The trade-off chart: the ML policy is cheaper at every service level.

## 3. Forecast explorer — 1.5 min

* Pick a high-volume item (top of the list). Show actuals, LightGBM forecast, the 80 %
  band and the moving-average baseline after the dashed "test" line.
* Scroll to **"Why this forecast?"** — SHAP: recent sales level, item identity, weekday,
  price. "The model is not a black box: every forecast is decomposed into drivers."
* Filter *Demand type = intermittent* — most retail series are mostly zeros, which is
  why we use a Tweedie loss and Croston/TSB baselines.

## 4. Inventory page — 1.5 min

* Table: forecast per day, **bias adjustment**, σ of lead-time error, safety stock,
  reorder point, EOQ order size — for every SKU-store.
* Click a row: the simulated inventory saw-tooth, reorder point line, purple orders, red
  lost sales. Toggle *Rule of thumb* to show the stock-outs it causes.

## 5. What-if — 1.5 min (the "wow" moment)

* Drag **lead time** from 7 → 14 days: ~2,000 policies are recomputed and 12 weeks
  re-simulated in about half a second; the ML advantage grows (≈ 1.5 %) because longer
  lead times make forecast accuracy matter more.
* Lower the **cost per order** to $2: orders become more frequent, again rewarding
  better forecasts. Press *Reset*.

## 6. Models page — 1 min

* Seven models, rolling-origin backtesting; champion chosen on **validation**, test shown
  for honesty. LightGBM: WRMSSE 0.834 vs 0.870 for the moving average.
* Global SHAP: the horizon-aware origin features dominate — the design decision that took
  LightGBM from "no better than a moving average" to clearly best.

## 7. Engineering — 1 min

```bash
uv run pytest -q          # 200+ tests, incl. real PostgreSQL, in ~30 s
```

* `mypy --strict`, ruff, 97 % coverage, GitHub Actions, Docker Compose, Alembic migrations.
* Highlight two tests: the **leakage test** (future values randomised → features unchanged)
  and the **what-if consistency test** (API re-simulation reproduces the pipeline exactly).

## Likely questions — short answers

| Question | Answer |
|---|---|
| Why is the ML gain only 0.8 % at 95 %? | Lot sizes are large (EOQ ≈ weeks of cover), so most cost is structural; forecasts change *when* to reorder. The gain grows with lead time, lower service targets and cheaper ordering — the what-if page shows it. Daily accuracy (4 % better WRMSSE) only partly transfers to 7-day cumulative error (8 % lower σ). |
| How do you know there is no leakage? | Lags ≥ horizon, origin features read only data at `t − h`, a test randomises all future targets and asserts identical features; policies may only use forecasts issued before each simulated day. |
| What went wrong during the project? | (1) A static baseline reorder point averaged forecasts from *future* windows — look-ahead bias that flattered the baseline. (2) A zero 28-day forecast collapsed EOQ to 1 unit, flattering ML. Both fixed; the comparison is now strictly like-for-like. |
| Why not deep learning? | M5 winners were LightGBM ensembles; on a laptop-sized subset trees train in ~100 s, explain exactly (TreeSHAP) and need no GPU. |
| Is the safety stock formula valid for intermittent demand? | It assumes roughly normal cumulative errors; cumulating over the lead time helps, and the bias term corrects systematic error. An empirical-quantile buffer is the natural extension (see limitations). |
| How would this go to production? | Schedule `run-all` weekly, publish to Postgres (already transactional + pruning), serve via the containers; MLflow tracks every run. |

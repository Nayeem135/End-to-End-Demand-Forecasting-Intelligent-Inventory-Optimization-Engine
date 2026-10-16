# Methodology

## 1. Data

**Source.** M5 Forecasting competition (Walmart): daily unit sales of 3,049 products in
10 stores across California, Texas and Wisconsin, 2011-01-29 to 2016-05-22 (1,941 days),
with weekly selling prices and a calendar of events and SNAP (food-stamp) days.

**Subset (configurable).** Default: state **CA** (4 stores), **500 items** drawn by
category-stratified random sampling (seed 42), last **730 days** → ~2,000 SKU-store
series, ~1.4 M rows. Everything is controlled by `DATA__STATE`, `DATA__N_ITEMS`,
`DATA__HISTORY_DAYS`.

**Preparation.** Wide → long (`unique_id, ds, y`), join calendar and prices, drop leading
zeros before an item's first sale (not real demand), forward-fill prices within a series,
drop series with fewer than 180 days of history before the first cut-off. A validator
enforces the contract: no nulls, non-negative demand, positive prices, unique keys,
contiguous dates.

**Demand classes** (Syntetos–Boylan): ADI (average inter-demand interval) and CV² of
non-zero sizes classify each series as smooth / erratic / intermittent / lumpy. Most M5
series are intermittent — which is why Croston/TSB are included as baselines and
LightGBM uses a Tweedie loss.

## 2. Evaluation protocol

```
... train ... | val 1 | val 2 | val 3 | test 1 | test 2 | test 3 |
                 28d     28d     28d     28d      28d      28d   (last date)
```

* **Rolling origin:** at each cut-off a *fresh* model is fitted on all data up to the
  cut-off and forecasts the next 28 days.
* **Validation folds** (3) select the champion and provide the forecast-error
  distribution used for safety stock.
* **Test windows** (3 × 28 = 84 days) are never used for any decision; they provide the
  reported accuracy and the demand on which inventory policies are simulated.

**Metrics.** MAE, RMSE, bias; **RMSSE** (RMSE scaled by the in-sample one-step naive
error, making series comparable) and **WRMSSE** (revenue-weighted RMSSE — the official
M5 metric) as the headline; **pinball loss** and **coverage** of the 80 % interval for
probabilistic quality.

## 3. Forecasting models

| Model | Idea | Why included |
|---|---|---|
| Naive | last value | sanity floor |
| Seasonal naive (7) | same weekday last week | weekly seasonality baseline |
| Moving average (28) | mean of last 28 days | what many retailers actually use; baseline of the inventory comparison |
| Croston (optimised) | separate smoothing of size and interval | classic intermittent-demand method |
| TSB | Croston variant with demand probability | handles obsolescence |
| ETS(A,N,A) | exponential smoothing with weekly seasonality | strong statistical benchmark |
| **LightGBM** | one global gradient-boosted model over all series | learns cross-series patterns, prices, events |

Statistical models use `statsforecast`; point methods get **conformal** prediction
intervals (5 calibration windows), ETS its likelihood intervals.

**LightGBM details.** Tweedie objective (variance power 1.1) for zero-inflated counts;
two extra quantile models give P10/P90. Features:

* calendar (weekday, month, day of month, week of year, events, SNAP),
* price (normalised price, 7-day price change, price relative to department),
* identifiers (item, department, category, store) as categoricals,
* **fixed lags** (28–364 days) and rolling mean/std/selling-rate shifted by 28 days,
* **origin features**: horizon *h* and statistics of demand at the forecast origin
  (last value, 7/28-day means, mean of the last four same-weekday values).

Fixed lags ≥ 28 days let one model forecast all 28 days directly (no recursion), but
discard the most recent month for short horizons. Origin features restore that
information *without leakage*: each training row receives a random horizon (seeded),
and the feature values are read at `t − h`. This single change improved validation
WRMSSE from 0.820 to 0.799 and made LightGBM's lead-time (7-day cumulative) error 8 %
smaller than the moving average's.

**Leakage tests.** For both lag and origin features, a unit test replaces all future
targets with random numbers and asserts that the features of the forecast window do not
change at all.

## 4. From forecast to inventory decision

Per SKU-store, with lead time *L*, review period *R* and an order placed at the end of
day *t* arriving at the start of day *t + L*, the inventory position must cover the
**protection period** *P = L + R − 1* days.

* **Forecast-error statistics.** From the validation folds we compute the *P*-day
  **cumulative** forecast error `E = Σ (y − ŷ)` over every window inside each fold and
  estimate its mean (**bias**) and standard deviation **σ**. Measuring cumulative errors
  directly captures error autocorrelation that `σ√P` would miss.
* **Reorder point:** `s_t = Σ_{k=1..P} ŷ_{t+k} + bias + z_α · σ`, with `z_α` the normal
  quantile of the target cycle service level α (default 95 %). The bias term corrects
  systematic under-/over-forecasting once rather than inflating it by `z`.
* **Order quantity (EOQ):** `Q = √(2 D K / H)` with the long-run annual demand *D* (last
  year of history), ordering cost *K* per order line and annual holding cost *H* = price
  × holding rate. Lot sizing is a long-run economic decision; using a 28-day forecast
  would let a zero forecast collapse *Q* to one unit.
* **Policy:** continuous-review **(s, nQ)** – when the position falls to `s_t` or below,
  order the smallest multiple of *Q* that lifts it above `s_t`.

## 5. Simulation (proof of value)

A vectorised, deterministic daily simulator replays the 84 test days for all series at
once: receive deliveries → serve demand (unmet demand is **lost**) → review and order →
pay holding cost on end-of-day stock. Costs: holding (25 %/year of price), ordering
($10 per order line), lost sales (1 × price). Everyone starts with 14 days of recent
average demand on hand.

Three policies face **identical demand, costs and starting stock**:

1. **Rule of thumb** — min/max: reorder at *L* days of recent average sales, fill up to
   14 days of cover; average refreshed monthly. No statistics.
2. **Moving average + safety stock** — the (s, nQ) policy of §4 fed with the
   moving-average forecast and its own error statistics.
3. **Forecast-driven (ML)** — the *identical* policy fed with the champion's daily
   forecast and its own error statistics.

Policies 2 and 3 differ **only in the forecast**, so their gap isolates the value of
forecast accuracy, while 1 → 2 measures the value of the inventory optimisation itself.

**Information rule (no look-ahead).** On day *t* a policy may only use the forecast
issued at the latest cut-off on or before *t*; beyond its own window a forecast is
continued by repeating its final week, never by peeking at the next window. Earlier
prototypes violated this subtly (a static reorder point averaged over all test windows),
which made the baseline look artificially good — the fix and the reasoning are part of
the project's findings.

**Sensitivity.** Service level is swept from 80 % to 99 % to trace each policy's
cost-versus-fill-rate frontier; the dashboard's what-if page lets any parameter be
changed live.

## 6. Explainability

`shap.TreeExplainer` computes exact TreeSHAP values for the LightGBM point model on the
last test window (5,000 sampled rows). With a log link, contributions are additive in
log-demand (`+0.2` ≈ +22 % on the forecast). Global importance = mean |SHAP|; per
series, the strongest drivers of its forecast are stored and shown in the dashboard.

## 7. Limitations and threats to validity

* One state, 500 items, 84 test days: results are indicative, not a full M5 benchmark.
* Lost-sales model with fixed, deterministic lead times; no supplier capacity, shelf
  life, minimum order quantities or shared-truck ordering costs.
* Cost parameters are assumptions (the what-if page exists precisely to show how
  conclusions depend on them).
* Safety stock assumes approximately normal cumulative errors; intermittent items are
  skewed. An empirical-quantile buffer is a natural extension.
* The test period (spring 2016) has slight demand growth that all models under-forecast;
  the bias correction mitigates but cannot fully remove this.

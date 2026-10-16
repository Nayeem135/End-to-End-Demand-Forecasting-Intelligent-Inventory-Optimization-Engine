// TypeScript mirror of the FastAPI response models (src/demand_engine/api/schemas.py).

export type PolicyName = 'rule_of_thumb' | 'ma_safety_stock' | 'forecast_driven'

export interface Headline {
  champion_test_wrmsse: number
  baseline_test_wrmsse: number
  champion_val_wrmsse: number
  savings_vs_rule_pct: number
  savings_vs_ma_pct: number
  fill_rate_forecast_driven: number
  fill_rate_rule_of_thumb: number
}

export interface RunSummary {
  run_id: string
  created_at: string
  state: string
  n_series: number
  n_items: number
  n_stores: number
  data_start: string
  data_end: string
  test_start: string
  champion: string
  baseline_model: string
  headline: Headline
  inventory: WhatIfParams
  mlflow_run_id: string | null
}

export interface ModelScore {
  split: 'validation' | 'test'
  model: string
  rank: number
  wrmsse: number
  rmsse: number
  mae: number
  rmse: number
  bias: number
  coverage: number
  pinball: number
  seconds: number
  is_champion: boolean
}

export interface Series {
  unique_id: string
  item_id: string
  dept_id: string
  cat_id: string
  store_id: string
  state_id: string
  avg_daily_sales: number
  avg_price: number
  demand_class: string
  champion_mae: number
}

export interface Page<T> {
  items: T[]
  total: number
  page: number
  page_size: number
}

export interface Policy {
  unique_id: string
  mean_daily_forecast: number
  forecast_bias: number
  sigma_protection: number
  safety_stock: number
  reorder_point: number
  order_quantity: number
  service_level: number
  lead_time_days: number
}

export interface PolicyRow extends Policy {
  item_id: string
  store_id: string
  cat_id: string
  demand_class: string
  avg_price: number
}

export interface SeriesResult {
  policy: PolicyName
  total_cost: number
  holding_cost: number
  ordering_cost: number
  stockout_cost: number
  fill_rate: number
  stockout_days: number
  avg_on_hand: number
  n_orders: number
}

export interface SeriesDetail {
  series: Series
  policy: Policy
  results: SeriesResult[]
}

export interface HistoryPoint {
  ds: string
  y: number
  sell_price: number
}

export interface ForecastPoint {
  ds: string
  yhat: number
  yhat_lo: number
  yhat_hi: number
}

export interface ForecastResponse {
  unique_id: string
  champion: string
  baseline_model: string
  test_start: string
  history: HistoryPoint[]
  forecasts: Record<string, ForecastPoint[]>
}

export interface DayState {
  ds: string
  demand: number
  on_hand: number
  inventory_position: number
  order: number
  sold: number
  lost: number
  reorder_point: number
}

export interface Trajectory {
  policy: PolicyName
  days: DayState[]
  summary: SeriesResult
}

export interface SeriesSimulation {
  unique_id: string
  trajectories: Trajectory[]
}

export interface Portfolio {
  policy: PolicyName
  total_cost: number
  holding_cost: number
  ordering_cost: number
  stockout_cost: number
  fill_rate: number
  stockout_day_rate: number
  avg_inventory_value: number
  n_orders: number
  savings_vs_baseline_pct: number
}

export interface SensitivityPoint {
  service_level: number
  policy: PolicyName
  total_cost: number
  holding_cost: number
  ordering_cost: number
  stockout_cost: number
  fill_rate: number
  avg_inventory_value: number
}

export interface WhatIfParams {
  service_level: number
  lead_time_days: number
  review_period_days: number
  ordering_cost: number
  holding_cost_rate: number
  stockout_cost_multiplier: number
}

export interface WhatIfResponse {
  params: WhatIfParams
  portfolio: Portfolio[]
  elapsed_ms: number
}

export interface FeatureImportance {
  feature: string
  mean_abs_shap: number
  gain: number | null
}

export interface ShapLocal {
  feature: string
  shap_value: number
  feature_value: number | null
}

export interface Filters {
  stores: string[]
  categories: string[]
  departments: string[]
  demand_classes: string[]
}

export interface Health {
  status: 'ok' | 'degraded'
  database: boolean
  latest_run: string | null
  version: string
}

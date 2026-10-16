import type { Portfolio, RunSummary, SensitivityPoint } from '../api/types'

export const RUN: RunSummary = {
  run_id: '20261016T120000',
  created_at: '2026-10-16T12:00:00Z',
  state: 'CA',
  n_series: 1961,
  n_items: 500,
  n_stores: 4,
  data_start: '2014-05-24',
  data_end: '2016-05-22',
  test_start: '2016-02-29',
  champion: 'lightgbm',
  baseline_model: 'moving_average',
  headline: {
    champion_test_wrmsse: 0.8,
    baseline_test_wrmsse: 0.9,
    champion_val_wrmsse: 0.78,
    savings_vs_rule_pct: 42.5,
    savings_vs_ma_pct: 6.2,
    fill_rate_forecast_driven: 0.985,
    fill_rate_rule_of_thumb: 0.88,
  },
  inventory: {
    service_level: 0.95,
    lead_time_days: 7,
    review_period_days: 1,
    ordering_cost: 10,
    holding_cost_rate: 0.25,
    stockout_cost_multiplier: 1,
  },
  mlflow_run_id: null,
}

const base = {
  holding_cost: 100,
  ordering_cost: 50,
  stockout_cost: 10,
  stockout_day_rate: 0.01,
  avg_inventory_value: 1000,
  n_orders: 10,
}

export const PORTFOLIO: Portfolio[] = [
  { ...base, policy: 'rule_of_thumb', total_cost: 1000, fill_rate: 0.88, savings_vs_baseline_pct: 0 },
  { ...base, policy: 'ma_safety_stock', total_cost: 640, fill_rate: 0.98, savings_vs_baseline_pct: 36 },
  { ...base, policy: 'forecast_driven', total_cost: 600, fill_rate: 0.985, savings_vs_baseline_pct: 40 },
]

export const SENSITIVITY: SensitivityPoint[] = [0.9, 0.95].flatMap((level) =>
  PORTFOLIO.map((p) => ({
    service_level: level,
    policy: p.policy,
    total_cost: p.total_cost,
    holding_cost: 1,
    ordering_cost: 1,
    stockout_cost: 1,
    fill_rate: p.fill_rate,
    avg_inventory_value: 1,
  })),
)

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>

/** A fetch replacement that answers by longest matching URL prefix (404 otherwise). */
export function mockFetch(routes: Record<string, unknown>): FetchLike {
  return async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
    const key = Object.keys(routes)
      .sort((a, b) => b.length - a.length)
      .find((k) => url.startsWith(k))
    if (!key) return new Response(JSON.stringify({ detail: `no mock for ${url}` }), { status: 404 })
    return new Response(JSON.stringify(routes[key]), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })
  }
}

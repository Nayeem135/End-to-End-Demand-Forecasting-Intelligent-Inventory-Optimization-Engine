// Consistent number / label formatting across the dashboard.
import type { PolicyName } from '../api/types'

const currency0 = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
})
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 })

/** Full currency, e.g. "$279,774" (tables, KPIs). */
export function formatCurrency(value: number): string {
  if (!Number.isFinite(value)) return '–'
  return currency0.format(value)
}

/** Compact currency for chart axes, e.g. "$280K". */
export function formatCurrencyCompact(value: number): string {
  if (!Number.isFinite(value)) return '–'
  return Math.abs(value) >= 10_000 ? `$${compact.format(value)}` : currency0.format(value)
}

export function formatPercent(fraction: number, digits = 1): string {
  if (!Number.isFinite(fraction)) return '–'
  return `${(fraction * 100).toFixed(digits)}%`
}

/** Format a value that is already in percent units (e.g. 12.3 -> "12.3%"). */
export function formatPct(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return '–'
  return `${value.toFixed(digits)}%`
}

export function formatNumber(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return '–'
  return value.toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: 0 })
}

export function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`)
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
}

export const POLICY_LABELS: Record<PolicyName, string> = {
  rule_of_thumb: 'Rule of thumb',
  ma_safety_stock: 'Moving avg + safety stock',
  forecast_driven: 'Forecast-driven (ML)',
}

export const POLICY_COLORS: Record<PolicyName, string> = {
  rule_of_thumb: '#f59e0b',
  ma_safety_stock: '#0ea5e9',
  forecast_driven: '#10b981',
}

const MODEL_LABELS: Record<string, string> = {
  naive: 'Naive',
  seasonal_naive: 'Seasonal naive',
  moving_average: 'Moving average',
  croston: 'Croston',
  tsb: 'TSB',
  ets: 'ETS',
  lightgbm: 'LightGBM',
}

export function modelLabel(name: string): string {
  return MODEL_LABELS[name] ?? name
}

/** Human-friendly feature names for SHAP charts, e.g. "roll_mean_28_7" -> "Mean sales (7d, lag 28)". */
export function featureLabel(name: string): string {
  const lag = /^lag_(\d+)$/.exec(name)
  if (lag) return `Sales ${lag[1]} days ago`
  const roll = /^roll_(mean|std)_(\d+)_(\d+)$/.exec(name)
  if (roll) return `${roll[1] === 'mean' ? 'Mean' : 'Volatility of'} sales (${roll[3]}d, lag ${roll[2]})`
  const origin = /^origin_mean_(\d+)$/.exec(name)
  if (origin) return `Recent mean sales (${origin[1]}d to forecast date)`
  const nz = /^nonzero_rate_(\d+)_(\d+)$/.exec(name)
  if (nz) return `Selling-days rate (${nz[2]}d, lag ${nz[1]})`
  const fixed: Record<string, string> = {
    horizon: 'Days ahead',
    origin_last: 'Sales on forecast date',
    origin_dow_mean: 'Same-weekday recent mean',
    item_id: 'Item identity',
    dept_id: 'Department',
    cat_id: 'Category',
    store_id: 'Store',
    event_type: 'Event type',
    wday: 'Day of week',
    month: 'Month',
    day_of_month: 'Day of month',
    week_of_year: 'Week of year',
    is_weekend: 'Weekend',
    is_event: 'Holiday / event',
    snap: 'SNAP day',
    sell_price: 'Price',
    price_norm: 'Price vs usual',
    price_change_7: 'Price change (7d)',
    price_rel_store_dept: 'Price vs department',
  }
  return fixed[name] ?? name
}

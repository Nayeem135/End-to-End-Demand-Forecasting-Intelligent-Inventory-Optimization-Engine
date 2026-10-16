import { describe, expect, it } from 'vitest'

import { featureLabel, formatCurrency, formatCurrencyCompact, formatPct, formatPercent, modelLabel } from './format'

describe('format helpers', () => {
  it('formats currency compactly for large values', () => {
    expect(formatCurrency(1234)).toBe('$1,234')
    expect(formatCurrency(2_500_000)).toBe('$2,500,000')
    expect(formatCurrencyCompact(2_500_000)).toBe('$2.5M')
    expect(formatCurrencyCompact(950)).toBe('$950')
    expect(formatCurrency(Number.NaN)).toBe('–')
  })

  it('formats fractions and percents', () => {
    expect(formatPercent(0.9512)).toBe('95.1%')
    expect(formatPct(12.345, 1)).toBe('12.3%')
    expect(formatPercent(Number.POSITIVE_INFINITY)).toBe('–')
  })

  it('labels models and features for humans', () => {
    expect(modelLabel('lightgbm')).toBe('LightGBM')
    expect(modelLabel('unknown_model')).toBe('unknown_model')
    expect(featureLabel('lag_28')).toBe('Sales 28 days ago')
    expect(featureLabel('roll_mean_28_7')).toBe('Mean sales (7d, lag 28)')
    expect(featureLabel('roll_std_28_56')).toBe('Volatility of sales (56d, lag 28)')
    expect(featureLabel('snap')).toBe('SNAP day')
    expect(featureLabel('origin_mean_28')).toBe('Recent mean sales (28d to forecast date)')
  })
})

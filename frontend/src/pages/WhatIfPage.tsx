import { useState } from 'react'

import { api } from '../api/client'
import type { Portfolio, WhatIfParams } from '../api/types'
import { CostBreakdownChart } from '../components/charts'
import { PageHeader } from '../components/PageHeader'
import { Card, ErrorState, KpiCard, Loading } from '../components/ui'
import { useApi, useDebounced } from '../hooks/useApi'
import { formatCurrency, formatPct, formatPercent, POLICY_COLORS, POLICY_LABELS } from '../lib/format'

interface SliderSpec {
  key: keyof WhatIfParams
  label: string
  min: number
  max: number
  step: number
  format: (v: number) => string
}

const SLIDERS: SliderSpec[] = [
  { key: 'service_level', label: 'Target service level', min: 0.8, max: 0.99, step: 0.01, format: (v) => formatPercent(v, 0) },
  { key: 'lead_time_days', label: 'Supplier lead time', min: 1, max: 21, step: 1, format: (v) => `${v} days` },
  { key: 'review_period_days', label: 'Review period', min: 1, max: 7, step: 1, format: (v) => `every ${v} day${v > 1 ? 's' : ''}` },
  { key: 'ordering_cost', label: 'Cost per order', min: 0, max: 50, step: 1, format: (v) => formatCurrency(v) },
  { key: 'holding_cost_rate', label: 'Holding cost (per year)', min: 0.05, max: 1, step: 0.05, format: (v) => formatPercent(v, 0) },
  { key: 'stockout_cost_multiplier', label: 'Lost-sale penalty', min: 0, max: 5, step: 0.25, format: (v) => `${v}× price` },
]

function find(portfolio: Portfolio[], policy: Portfolio['policy']): Portfolio | undefined {
  return portfolio.find((p) => p.policy === policy)
}

export function WhatIfPage() {
  const run = useApi(() => api.overview(), [])
  const [edited, setEdited] = useState<WhatIfParams | null>(null)
  const params = edited ?? run.data?.inventory ?? null
  const debounced = useDebounced(params, 350)
  const result = useApi(() => (debounced ? api.whatIf(debounced) : Promise.resolve(null)), [debounced], {
    keepPrevious: true,
  })

  if (run.loading || !params) return <Loading />
  if (run.error || !run.data) return <ErrorState message={run.error ?? 'No data'} />
  const portfolio = result.data?.portfolio ?? []
  const fd = find(portfolio, 'forecast_driven')
  const ma = find(portfolio, 'ma_safety_stock')
  const mlGain = fd && ma && ma.total_cost > 0 ? (100 * (ma.total_cost - fd.total_cost)) / ma.total_cost : NaN

  return (
    <>
      <PageHeader
        title="What-if simulator"
        description="Change the business assumptions and the engine recomputes safety stock, reorder points and order sizes for every SKU-store, then replays the whole test period live."
      >
        <button
          type="button"
          onClick={() => setEdited(null)}
          className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm hover:bg-slate-50"
        >
          Reset to run defaults
        </button>
      </PageHeader>
      <div className="grid gap-6 lg:grid-cols-3">
        <Card title="Assumptions" className="lg:col-span-1">
          <div className="space-y-5">
            {SLIDERS.map((s) => (
              <label key={s.key} className="block">
                <span className="flex justify-between text-sm">
                  <span className="font-medium text-slate-700">{s.label}</span>
                  <span className="tabular-nums text-slate-900">{s.format(params[s.key])}</span>
                </span>
                <input
                  aria-label={s.label}
                  type="range"
                  className="mt-2 w-full accent-indigo-600"
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={params[s.key]}
                  onChange={(e) => setEdited({ ...params, [s.key]: Number(e.target.value) })}
                />
              </label>
            ))}
          </div>
        </Card>
        <div className="space-y-6 lg:col-span-2">
          {result.error && <ErrorState message={result.error} />}
          <div className="grid gap-4 sm:grid-cols-3">
            <KpiCard
              label="Saving vs rule-of-thumb"
              value={fd ? formatPct(fd.savings_vs_baseline_pct) : '…'}
              tone={fd && fd.savings_vs_baseline_pct > 0 ? 'good' : 'bad'}
            />
            <KpiCard label="Extra saving from ML" value={formatPct(mlGain)} tone={mlGain > 0 ? 'good' : 'bad'} />
            <KpiCard
              label="Forecast-driven fill rate"
              value={fd ? formatPercent(fd.fill_rate) : '…'}
              hint={result.data ? `simulated in ${result.data.elapsed_ms.toFixed(0)} ms` : 'simulating…'}
            />
          </div>
          <Card title="Cost by policy under these assumptions">
            {result.loading && !result.data ? <Loading label="Simulating…" /> : <CostBreakdownChart portfolio={portfolio} />}
          </Card>
          <Card title="Detail">
            <table className="w-full text-sm">
              <thead className="text-left text-xs uppercase text-slate-500">
                <tr>
                  <th className="pb-2">Policy</th>
                  <th className="pb-2 text-right">Total cost</th>
                  <th className="pb-2 text-right">Fill rate</th>
                  <th className="pb-2 text-right">Avg stock value</th>
                  <th className="pb-2 text-right">Orders</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {portfolio.map((p) => (
                  <tr key={p.policy}>
                    <td className="py-2">
                      <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: POLICY_COLORS[p.policy] }} />
                      {POLICY_LABELS[p.policy]}
                    </td>
                    <td className="py-2 text-right tabular-nums">{formatCurrency(p.total_cost)}</td>
                    <td className="py-2 text-right tabular-nums">{formatPercent(p.fill_rate)}</td>
                    <td className="py-2 text-right tabular-nums">{formatCurrency(p.avg_inventory_value)}</td>
                    <td className="py-2 text-right tabular-nums">{p.n_orders.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      </div>
    </>
  )
}

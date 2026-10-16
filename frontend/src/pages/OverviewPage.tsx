import { useState } from 'react'

import { api } from '../api/client'
import type { Portfolio, RunSummary, SensitivityPoint } from '../api/types'
import { CostBreakdownChart, SensitivityChart } from '../components/charts'
import { PageHeader } from '../components/PageHeader'
import { Card, ErrorState, KpiCard, Loading } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { formatCurrency, formatPct, formatPercent, modelLabel, POLICY_COLORS, POLICY_LABELS } from '../lib/format'

interface OverviewData {
  run: RunSummary
  portfolio: Portfolio[]
  sensitivity: SensitivityPoint[]
}

async function loadOverview(): Promise<OverviewData> {
  const [run, portfolio, sensitivity] = await Promise.all([api.overview(), api.portfolio(), api.sensitivity()])
  return { run, portfolio, sensitivity }
}

const STEPS = [
  ['1', 'Ingest', 'M5 Walmart sales, prices & calendar, validated and subset'],
  ['2', 'Forecast', '7 models backtested on rolling origins; best picked on validation'],
  ['3', 'Optimise', 'Safety stock, reorder point & EOQ from measured forecast error'],
  ['4', 'Simulate', 'Day-by-day replay on unseen demand proves the cost saving'],
]

export function OverviewPage() {
  const state = useApi(loadOverview, [])
  const [metric, setMetric] = useState<'total_cost' | 'fill_rate'>('total_cost')

  if (state.loading) return <Loading />
  if (state.error || !state.data) return <ErrorState message={state.error ?? 'No data'} />
  const { run, portfolio, sensitivity } = state.data
  const h = run.headline
  const accuracyGain = (1 - h.champion_test_wrmsse / h.baseline_test_wrmsse) * 100
  const ordered = [...portfolio].sort((a, b) => b.total_cost - a.total_cost)

  return (
    <>
      <PageHeader
        title="Results overview"
        description={`${run.n_series.toLocaleString()} SKU-store series (${run.n_items} items × ${run.n_stores} ${run.state} stores). Policies are simulated on ${run.test_start} onwards — demand none of the models ever saw.`}
      />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <KpiCard
          label="Inventory cost saving"
          value={formatPct(h.savings_vs_rule_pct)}
          hint="Forecast-driven policy vs rule-of-thumb"
          tone={h.savings_vs_rule_pct > 0 ? 'good' : 'bad'}
        />
        <KpiCard
          label="Extra saving from ML"
          value={formatPct(h.savings_vs_ma_pct)}
          hint={`${modelLabel(run.champion)} vs moving-average forecast`}
          tone={h.savings_vs_ma_pct > 0 ? 'good' : 'bad'}
        />
        <KpiCard
          label="Fill rate"
          value={formatPercent(h.fill_rate_forecast_driven)}
          hint={`was ${formatPercent(h.fill_rate_rule_of_thumb)} with rule-of-thumb`}
          tone="good"
        />
        <KpiCard
          label="Forecast accuracy gain"
          value={formatPct(accuracyGain)}
          hint={`WRMSSE ${h.champion_test_wrmsse.toFixed(3)} vs ${h.baseline_test_wrmsse.toFixed(3)} (${modelLabel(run.baseline_model)})`}
          tone={accuracyGain > 0 ? 'good' : 'bad'}
        />
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-5">
        <Card className="lg:col-span-3" title="Where the money goes" subtitle="Total simulated cost by component and policy">
          <CostBreakdownChart portfolio={ordered} />
        </Card>
        <Card className="lg:col-span-2" title="Policy scorecard" subtitle="Whole portfolio over the test period">
          <table className="w-full text-sm">
            <thead className="text-left text-xs uppercase text-slate-500">
              <tr>
                <th className="pb-2">Policy</th>
                <th className="pb-2 text-right">Cost</th>
                <th className="pb-2 text-right">Fill rate</th>
                <th className="pb-2 text-right">Orders</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {ordered.map((p) => (
                <tr key={p.policy}>
                  <td className="py-2">
                    <span className="mr-2 inline-block h-2.5 w-2.5 rounded-full" style={{ background: POLICY_COLORS[p.policy] }} />
                    {POLICY_LABELS[p.policy]}
                  </td>
                  <td className="py-2 text-right tabular-nums">{formatCurrency(p.total_cost)}</td>
                  <td className="py-2 text-right tabular-nums">{formatPercent(p.fill_rate)}</td>
                  <td className="py-2 text-right tabular-nums">{p.n_orders.toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-4 text-xs text-slate-500">
            Assumptions: lead time {run.inventory.lead_time_days} days, target service level{' '}
            {formatPercent(run.inventory.service_level, 0)}, holding {formatPercent(run.inventory.holding_cost_rate, 0)} of
            price per year, {formatCurrency(run.inventory.ordering_cost)} per order, lost sales cost{' '}
            {run.inventory.stockout_cost_multiplier}× price. Identical demand and costs for every policy.
          </p>
        </Card>
      </div>

      <Card
        className="mt-6"
        title="Cost vs service trade-off"
        subtitle={`The two optimised policies across target service levels. Rule-of-thumb ignores the target: ${formatCurrency(ordered[0]?.total_cost ?? Number.NaN)} at ${formatPercent(ordered[0]?.fill_rate ?? Number.NaN)} fill rate.`}
        actions={
          <div className="flex rounded-lg border border-slate-200 p-0.5 text-sm">
            {(['total_cost', 'fill_rate'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMetric(m)}
                className={`rounded-md px-3 py-1 ${metric === m ? 'bg-slate-900 text-white' : 'text-slate-600'}`}
              >
                {m === 'total_cost' ? 'Total cost' : 'Fill rate'}
              </button>
            ))}
          </div>
        }
      >
        <SensitivityChart points={sensitivity} metric={metric} exclude={['rule_of_thumb']} />
      </Card>

      <div className="mt-6 grid gap-4 md:grid-cols-4">
        {STEPS.map(([n, title, text]) => (
          <div key={n} className="rounded-xl border border-dashed border-slate-300 p-4">
            <p className="text-xs font-semibold text-indigo-600">STEP {n}</p>
            <p className="mt-1 font-medium text-slate-900">{title}</p>
            <p className="mt-1 text-sm text-slate-500">{text}</p>
          </div>
        ))}
      </div>
    </>
  )
}

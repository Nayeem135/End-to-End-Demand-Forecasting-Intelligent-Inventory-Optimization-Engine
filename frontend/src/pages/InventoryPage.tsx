import { useState } from 'react'

import { api } from '../api/client'
import type { PolicyName, PolicyRow } from '../api/types'
import { InventoryChart } from '../components/charts'
import { PageHeader } from '../components/PageHeader'
import { SeriesFilters } from '../components/SeriesFilters'
import { Card, Empty, ErrorState, Loading, Pagination } from '../components/ui'
import { useApi, useDebounced } from '../hooks/useApi'
import { EMPTY_FILTERS, type FilterState } from '../lib/filters'
import { formatCurrency, formatNumber, formatPercent, POLICY_COLORS, POLICY_LABELS } from '../lib/format'

const PAGE_SIZE = 10

function SimulationPanel({ row }: { row: PolicyRow }) {
  const sim = useApi(() => api.simulation(row.unique_id), [row.unique_id])
  const [policy, setPolicy] = useState<PolicyName>('forecast_driven')
  if (sim.loading) return <Loading label="Simulating…" />
  if (sim.error || !sim.data) return <ErrorState message={sim.error ?? 'No data'} />
  const current = sim.data.trajectories.find((t) => t.policy === policy) ?? sim.data.trajectories[0]
  if (!current) return <Empty message="No simulation available." />

  return (
    <Card
      title={`Simulated inventory · ${row.item_id} @ ${row.store_id}`}
      subtitle="Replay of the test period: daily on-hand stock, reorder point, orders and lost sales"
      actions={
        <div className="flex flex-wrap gap-1 rounded-lg border border-slate-200 p-0.5 text-sm">
          {sim.data.trajectories.map((t) => (
            <button
              key={t.policy}
              type="button"
              onClick={() => setPolicy(t.policy)}
              className={`rounded-md px-3 py-1 ${t.policy === current.policy ? 'text-white' : 'text-slate-600'}`}
              style={t.policy === current.policy ? { background: POLICY_COLORS[t.policy] } : undefined}
            >
              {POLICY_LABELS[t.policy]}
            </button>
          ))}
        </div>
      }
    >
      <InventoryChart trajectory={current} />
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        {sim.data.trajectories.map((t) => (
          <div key={t.policy} className="rounded-lg border border-slate-200 p-3">
            <p className="text-xs font-medium" style={{ color: POLICY_COLORS[t.policy] }}>
              {POLICY_LABELS[t.policy]}
            </p>
            <p className="mt-1 font-semibold tabular-nums">{formatCurrency(t.summary.total_cost)}</p>
            <p className="text-xs text-slate-500">
              fill {formatPercent(t.summary.fill_rate)} · {t.summary.n_orders} orders · {t.summary.stockout_days} stock-out days
            </p>
          </div>
        ))}
      </div>
    </Card>
  )
}

export function InventoryPage() {
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS)
  const [page, setPage] = useState(1)
  const [picked, setPicked] = useState<PolicyRow | null>(null)
  const debounced = useDebounced(filters, 300)
  const options = useApi(() => api.filters(), [])
  const list = useApi(() => api.policies({ ...debounced, page, page_size: PAGE_SIZE }), [debounced, page])
  const selected = picked ?? list.data?.items[0] ?? null
  const onFilters = (f: FilterState) => {
    setFilters(f)
    setPage(1)
  }

  return (
    <>
      <PageHeader
        title="Inventory recommendations"
        description="For every SKU-store: a reorder point that follows the daily forecast, a buffer from the measured bias and spread of lead-time forecast errors, and an EOQ order size."
      />
      <Card>
        <SeriesFilters filters={filters} options={options.data} onChange={onFilters} />
        <div className="mt-4 overflow-x-auto">
          {list.loading && <Loading />}
          {list.error && <ErrorState message={list.error} />}
          {list.data && list.data.items.length === 0 && <Empty message="No series match these filters." />}
          {list.data && list.data.items.length > 0 && (
            <table className="w-full min-w-[760px] text-sm">
              <thead className="text-left text-xs uppercase text-slate-500">
                <tr>
                  <th className="pb-2">SKU · store</th>
                  <th className="pb-2">Type</th>
                  <th className="pb-2 text-right">Forecast /day</th>
                  <th className="pb-2 text-right" title="Mean under-forecast over the protection period (added to the reorder point)">Bias adj.</th>
                  <th className="pb-2 text-right" title="Std of cumulative forecast error over the protection period"><span className="normal-case">σ</span> lead time</th>
                  <th className="pb-2 text-right">Safety stock</th>
                  <th className="pb-2 text-right">Reorder point</th>
                  <th className="pb-2 text-right">Order qty</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {list.data.items.map((r) => (
                  <tr
                    key={r.unique_id}
                    onClick={() => setPicked(r)}
                    className={`cursor-pointer hover:bg-slate-50 ${selected?.unique_id === r.unique_id ? 'bg-indigo-50' : ''}`}
                  >
                    <td className="py-2">
                      <span className="font-medium text-slate-900">{r.item_id}</span>
                      <span className="ml-2 text-slate-500">{r.store_id}</span>
                    </td>
                    <td className="py-2 text-slate-500">{r.demand_class}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(r.mean_daily_forecast, 2)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(r.forecast_bias, 1)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(r.sigma_protection, 1)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(r.safety_stock, 1)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(r.reorder_point, 1)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(r.order_quantity, 0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {list.data && <Pagination page={page} pageSize={PAGE_SIZE} total={list.data.total} onPage={setPage} />}
        </div>
      </Card>
      <div className="mt-6">{selected && <SimulationPanel row={selected} />}</div>
    </>
  )
}

import { useState } from 'react'

import { api } from '../api/client'
import type { Page, Series } from '../api/types'
import { ForecastChart, HorizontalBarChart } from '../components/charts'
import { PageHeader } from '../components/PageHeader'
import { SeriesFilters } from '../components/SeriesFilters'
import { Badge, Card, Empty, ErrorState, Loading, Pagination } from '../components/ui'
import { useApi, useDebounced, type ApiState } from '../hooks/useApi'
import { EMPTY_FILTERS, type FilterState } from '../lib/filters'
import { featureLabel, formatCurrency, formatNumber, modelLabel } from '../lib/format'

const PAGE_SIZE = 12

interface SeriesListProps {
  list: ApiState<Page<Series>>
  filters: FilterState
  onFilters: (f: FilterState) => void
  page: number
  onPage: (p: number) => void
  selected: string | null
  onSelect: (s: Series) => void
}

function SeriesList({ list, filters, onFilters, page, onPage, selected, onSelect }: SeriesListProps) {
  const options = useApi(() => api.filters(), [])
  return (
    <Card title="SKU-store series" subtitle="Sorted by average daily sales">
      <SeriesFilters filters={filters} options={options.data} onChange={onFilters} />
      <div className="mt-4">
        {list.loading && <Loading />}
        {list.error && <ErrorState message={list.error} />}
        {list.data && list.data.items.length === 0 && <Empty message="No series match these filters." />}
        {list.data && list.data.items.length > 0 && (
          <>
            <ul className="divide-y divide-slate-100">
              {list.data.items.map((s) => (
                <li key={s.unique_id}>
                  <button
                    type="button"
                    onClick={() => onSelect(s)}
                    className={`flex w-full items-center justify-between gap-2 rounded-md px-2 py-2 text-left text-sm hover:bg-slate-50 ${
                      selected === s.unique_id ? 'bg-indigo-50 ring-1 ring-indigo-200' : ''
                    }`}
                  >
                    <span>
                      <span className="font-medium text-slate-900">{s.item_id}</span>
                      <span className="ml-2 text-slate-500">{s.store_id}</span>
                    </span>
                    <span className="text-xs text-slate-500 tabular-nums">{formatNumber(s.avg_daily_sales, 1)}/day</span>
                  </button>
                </li>
              ))}
            </ul>
            <Pagination page={page} pageSize={PAGE_SIZE} total={list.data.total} onPage={onPage} />
          </>
        )}
      </div>
    </Card>
  )
}

function SeriesForecast({ series }: { series: Series }) {
  const fc = useApi(() => api.forecast(series.unique_id), [series.unique_id])
  const shap = useApi(() => api.explanation(series.unique_id), [series.unique_id])

  return (
    <div className="space-y-6">
      <Card
        title={`${series.item_id} · ${series.store_id}`}
        subtitle="Last 16 weeks of actual sales, then 12 weeks of out-of-sample forecasts"
        actions={
          <div className="flex flex-wrap gap-2">
            <Badge color="indigo">{series.cat_id}</Badge>
            <Badge color="amber">{series.demand_class}</Badge>
            <Badge>{formatCurrency(series.avg_price)} avg price</Badge>
          </div>
        }
      >
        {fc.loading && <Loading />}
        {fc.error && <ErrorState message={fc.error} />}
        {fc.data && <ForecastChart data={fc.data} />}
        {fc.data && (
          <p className="mt-2 text-xs text-slate-500">
            Champion: {modelLabel(fc.data.champion)} (selected on validation folds). Mean absolute error on the test period:{' '}
            {formatNumber(series.champion_mae, 2)} units/day.
          </p>
        )}
      </Card>
      <Card title="Why this forecast?" subtitle="SHAP contributions (log-demand). Green raises the forecast, red lowers it.">
        {shap.loading && <Loading />}
        {shap.error && <ErrorState message={shap.error} />}
        {shap.data && shap.data.length === 0 && <Empty message="No SHAP explanation stored for this series." />}
        {shap.data && shap.data.length > 0 && (
          <HorizontalBarChart signed items={shap.data.map((d) => ({ label: featureLabel(d.feature), value: d.shap_value }))} />
        )}
      </Card>
    </div>
  )
}

export function ForecastsPage() {
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS)
  const [page, setPage] = useState(1)
  const [picked, setPicked] = useState<Series | null>(null)
  const debounced = useDebounced(filters, 300)
  const list = useApi(() => api.series({ ...debounced, page, page_size: PAGE_SIZE }), [debounced, page])
  // Until the user picks a series, show the top one of the current list.
  const selected = picked ?? list.data?.items[0] ?? null
  const onFilters = (f: FilterState) => {
    setFilters(f)
    setPage(1)
  }

  return (
    <>
      <PageHeader
        title="Forecast explorer"
        description="Pick any SKU-store to compare the champion model's forecast and 80% prediction interval against actual sales, and see which features drove it."
      />
      <div className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <SeriesList
            list={list}
            filters={filters}
            onFilters={onFilters}
            page={page}
            onPage={setPage}
            selected={selected?.unique_id ?? null}
            onSelect={setPicked}
          />
        </div>
        <div className="lg:col-span-2">
          {selected ? <SeriesForecast series={selected} /> : list.loading ? <Loading /> : <Empty message="No series selected." />}
        </div>
      </div>
    </>
  )
}

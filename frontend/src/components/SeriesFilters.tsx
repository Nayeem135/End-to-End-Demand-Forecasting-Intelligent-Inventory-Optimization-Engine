import type { Filters } from '../api/types'
import type { FilterState } from '../lib/filters'
import { Select } from './ui'

export function SeriesFilters({
  filters,
  options,
  onChange,
}: {
  filters: FilterState
  options: Filters | null
  onChange: (next: FilterState) => void
}) {
  return (
    <div className="flex flex-wrap items-end gap-3">
      <label className="flex min-w-48 flex-1 flex-col gap-1 text-xs font-medium text-slate-600">
        Search SKU
        <input
          type="search"
          placeholder="e.g. FOODS_3_090"
          className="rounded-md border border-slate-300 px-2 py-1.5 text-sm focus:border-indigo-500 focus:outline-none"
          value={filters.search}
          onChange={(e) => onChange({ ...filters, search: e.target.value })}
        />
      </label>
      <Select label="Store" value={filters.store_id} options={options?.stores ?? []} onChange={(v) => onChange({ ...filters, store_id: v })} />
      <Select label="Category" value={filters.cat_id} options={options?.categories ?? []} onChange={(v) => onChange({ ...filters, cat_id: v })} />
      <Select
        label="Demand type"
        value={filters.demand_class}
        options={options?.demand_classes ?? []}
        onChange={(v) => onChange({ ...filters, demand_class: v })}
      />
    </div>
  )
}

// Recharts visualisations used across the dashboard.
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { ForecastResponse, ModelScore, PolicyName, Portfolio, SensitivityPoint, Trajectory } from '../api/types'
import {
  formatCurrency,
  formatCurrencyCompact,
  formatDate,
  formatNumber,
  formatPercent,
  modelLabel,
  POLICY_COLORS,
  POLICY_LABELS,
} from '../lib/format'

const AXIS = { fontSize: 12, fill: '#64748b' }

export function CostBreakdownChart({ portfolio }: { portfolio: Portfolio[] }) {
  const data = portfolio.map((p) => ({
    name: POLICY_LABELS[p.policy],
    Holding: p.holding_cost,
    Ordering: p.ordering_cost,
    Stockout: p.stockout_cost,
  }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ left: 8, right: 8 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={AXIS} />
        <YAxis tick={AXIS} tickFormatter={(v: number) => formatCurrencyCompact(v)} width={70} />
        <Tooltip formatter={(v) => formatCurrency(Number(v))} />
        <Legend />
        <Bar dataKey="Holding" stackId="c" fill="#6366f1" />
        <Bar dataKey="Ordering" stackId="c" fill="#94a3b8" />
        <Bar dataKey="Stockout" stackId="c" fill="#f43f5e" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function SensitivityChart({
  points,
  metric,
  exclude = [],
}: {
  points: SensitivityPoint[]
  metric: 'total_cost' | 'fill_rate'
  exclude?: PolicyName[]
}) {
  const levels = [...new Set(points.map((p) => p.service_level))].sort((a, b) => a - b)
  const policies = [...new Set(points.map((p) => p.policy))].filter((p) => !exclude.includes(p))
  const data = levels.map((level) => {
    const row: Record<string, number> = { level }
    for (const p of points.filter((x) => x.service_level === level)) row[p.policy] = p[metric]
    return row
  })
  const axisFmt = metric === 'total_cost' ? formatCurrencyCompact : (v: number) => formatPercent(v)
  const fmt = metric === 'total_cost' ? formatCurrency : (v: number) => formatPercent(v)
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ left: 8, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="level" tick={AXIS} tickFormatter={(v: number) => formatPercent(v, 0)} />
        <YAxis tick={AXIS} tickFormatter={axisFmt} width={70} domain={['auto', 'auto']} />
        <Tooltip formatter={(v) => fmt(Number(v))} labelFormatter={(l) => `Target service level ${formatPercent(Number(l), 0)}`} />
        <Legend formatter={(v: string) => POLICY_LABELS[v as PolicyName] ?? v} />
        {policies.map((p) => (
          <Line key={p} type="monotone" dataKey={p} stroke={POLICY_COLORS[p]} strokeWidth={2.5} dot={{ r: 3 }} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

interface ForecastRow {
  ds: string
  actual?: number
  champion?: number
  band?: [number, number]
  baseline?: number
}

export function ForecastChart({ data }: { data: ForecastResponse }) {
  const rows = new Map<string, ForecastRow>()
  const get = (ds: string): ForecastRow => {
    const existing = rows.get(ds)
    if (existing) return existing
    const created: ForecastRow = { ds }
    rows.set(ds, created)
    return created
  }
  for (const h of data.history) get(h.ds).actual = h.y
  for (const f of data.forecasts[data.champion] ?? []) {
    const row = get(f.ds)
    row.champion = f.yhat
    row.band = [f.yhat_lo, f.yhat_hi]
  }
  if (data.baseline_model !== data.champion) {
    for (const f of data.forecasts[data.baseline_model] ?? []) get(f.ds).baseline = f.yhat
  }
  const series = [...rows.values()].sort((a, b) => a.ds.localeCompare(b.ds))
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ComposedChart data={series} margin={{ left: 0, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="ds" tick={AXIS} tickFormatter={formatDate} minTickGap={32} />
        <YAxis tick={AXIS} width={40} allowDecimals={false} />
        <Tooltip
          labelFormatter={(l) => formatDate(String(l))}
          formatter={(v, name) => {
            if (Array.isArray(v)) return [`${formatNumber(Number(v[0]), 1)} – ${formatNumber(Number(v[1]), 1)}`, name]
            return [formatNumber(Number(v), 2), name]
          }}
        />
        <Legend />
        <ReferenceLine x={data.test_start} stroke="#475569" strokeDasharray="4 4" label={{ value: 'test →', position: 'insideTopRight', fill: '#475569', fontSize: 12 }} />
        <Area dataKey="band" name="80% interval" stroke="none" fill="#10b981" fillOpacity={0.15} />
        <Line dataKey="actual" name="Actual sales" stroke="#334155" dot={false} strokeWidth={1.5} />
        <Line dataKey="champion" name={`${modelLabel(data.champion)} forecast`} stroke="#10b981" dot={false} strokeWidth={2.5} />
        {data.baseline_model !== data.champion && (
          <Line dataKey="baseline" name={`${modelLabel(data.baseline_model)} forecast`} stroke="#0ea5e9" dot={false} strokeDasharray="5 4" strokeWidth={1.5} />
        )}
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function InventoryChart({ trajectory }: { trajectory: Trajectory }) {
  const color = POLICY_COLORS[trajectory.policy]
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ComposedChart data={trajectory.days} margin={{ left: 0, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="ds" tick={AXIS} tickFormatter={formatDate} minTickGap={32} />
        <YAxis tick={AXIS} width={44} />
        <Tooltip labelFormatter={(l) => formatDate(String(l))} formatter={(v, name) => [formatNumber(Number(v), 1), name]} />
        <Legend />
        <Area type="stepAfter" dataKey="on_hand" name="On hand" stroke={color} fill={color} fillOpacity={0.2} strokeWidth={2} />
        <Line type="stepAfter" dataKey="reorder_point" name="Reorder point" stroke="#475569" strokeDasharray="5 4" dot={false} />
        <Bar dataKey="order" name="Order placed" fill="#6366f1" barSize={6} />
        <Bar dataKey="lost" name="Lost sales" fill="#f43f5e" barSize={6} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

export function HorizontalBarChart({
  items,
  valueFormatter = (v: number) => formatNumber(v, 3),
  signed = false,
  height,
}: {
  items: { label: string; value: number }[]
  valueFormatter?: (v: number) => string
  signed?: boolean
  height?: number
}) {
  return (
    <ResponsiveContainer width="100%" height={height ?? Math.max(160, items.length * 30 + 40)}>
      <BarChart data={items} layout="vertical" margin={{ left: 8, right: 24 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} />
        <XAxis type="number" tick={AXIS} tickFormatter={valueFormatter} />
        <YAxis type="category" dataKey="label" tick={AXIS} width={190} />
        <Tooltip formatter={(v) => valueFormatter(Number(v))} />
        {signed && <ReferenceLine x={0} stroke="#94a3b8" />}
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {items.map((it) => (
            <Cell key={it.label} fill={signed ? (it.value >= 0 ? '#10b981' : '#f43f5e') : '#6366f1'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

export function LeaderboardChart({ scores }: { scores: ModelScore[] }) {
  const data = scores.map((s) => ({ name: modelLabel(s.model), wrmsse: s.wrmsse, champion: s.is_champion }))
  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={data} margin={{ left: 0, right: 16 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={AXIS} interval={0} angle={-30} textAnchor="end" height={60} />
        <YAxis tick={AXIS} width={44} domain={[0, 'auto']} />
        <Tooltip formatter={(v) => formatNumber(Number(v), 4)} />
        <Bar dataKey="wrmsse" name="WRMSSE (lower is better)" radius={[4, 4, 0, 0]}>
          {data.map((d) => (
            <Cell key={d.name} fill={d.champion ? '#10b981' : '#94a3b8'} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

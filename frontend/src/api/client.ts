// Typed HTTP client for the FastAPI backend.
import type {
  FeatureImportance,
  Filters,
  ForecastResponse,
  Health,
  ModelScore,
  Page,
  PolicyRow,
  Portfolio,
  RunSummary,
  SensitivityPoint,
  Series,
  SeriesDetail,
  SeriesSimulation,
  ShapLocal,
  WhatIfParams,
  WhatIfResponse,
} from './types'

const API_BASE: string = import.meta.env.VITE_API_BASE ?? ''

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

export type QueryValue = string | number | boolean | null | undefined

/** Build `?a=1&b=x`, skipping null/undefined/empty values. */
export function buildQuery(params: Record<string, QueryValue>): string {
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, String(value))
  }
  const text = search.toString()
  return text ? `?${text}` : ''
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!response.ok) {
    let detail = response.statusText
    try {
      const body: unknown = await response.json()
      if (body && typeof body === 'object' && 'detail' in body) {
        const d = (body as { detail: unknown }).detail
        detail = typeof d === 'string' ? d : JSON.stringify(d)
      }
    } catch {
      // keep statusText when the body is not JSON
    }
    throw new ApiError(response.status, detail)
  }
  return (await response.json()) as T
}

export interface SeriesFilter {
  search?: string
  store_id?: string
  cat_id?: string
  demand_class?: string
  sort?: 'unique_id' | 'avg_daily_sales' | 'avg_price' | 'champion_mae'
  descending?: boolean
  page?: number
  page_size?: number
}

export const api = {
  health: () => request<Health>('/api/health'),
  overview: () => request<RunSummary>('/api/overview'),
  leaderboard: (split: 'validation' | 'test') =>
    request<ModelScore[]>(`/api/models/leaderboard${buildQuery({ split })}`),
  filters: () => request<Filters>('/api/filters'),
  series: (filter: SeriesFilter) =>
    request<Page<Series>>(`/api/series${buildQuery({ ...filter })}`),
  seriesDetail: (id: string) => request<SeriesDetail>(`/api/series/${encodeURIComponent(id)}`),
  forecast: (id: string) =>
    request<ForecastResponse>(`/api/series/${encodeURIComponent(id)}/forecast`),
  simulation: (id: string) =>
    request<SeriesSimulation>(`/api/series/${encodeURIComponent(id)}/simulation`),
  explanation: (id: string) =>
    request<ShapLocal[]>(`/api/series/${encodeURIComponent(id)}/explanation`),
  globalExplanation: (limit = 15) =>
    request<FeatureImportance[]>(`/api/explainability/global${buildQuery({ limit })}`),
  policies: (filter: SeriesFilter) =>
    request<Page<PolicyRow>>(`/api/inventory/policies${buildQuery({ ...filter })}`),
  portfolio: () => request<Portfolio[]>('/api/simulation/portfolio'),
  sensitivity: () => request<SensitivityPoint[]>('/api/simulation/sensitivity'),
  whatIf: (params: Partial<WhatIfParams>) =>
    request<WhatIfResponse>('/api/simulation/what-if', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
}

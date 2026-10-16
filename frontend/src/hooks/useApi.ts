import { useEffect, useEffectEvent, useState } from 'react'

export interface ApiState<T> {
  data: T | null
  error: string | null
  loading: boolean
}

interface Settled<T> {
  key: string
  data: T | null
  error: string | null
}

/**
 * Run an async loader whenever the serialised `deps` change.
 *
 * Loading is derived (the last settled request has a different key), so no state
 * is set synchronously inside the effect, and stale responses are ignored so
 * fast filter changes never show out-of-order data. With `keepPrevious`, the
 * previous data stays visible while a new request is in flight.
 */
export function useApi<T>(
  loader: () => Promise<T>,
  deps: readonly unknown[],
  options: { keepPrevious?: boolean } = {},
): ApiState<T> {
  const key = JSON.stringify(deps)
  const [settled, setSettled] = useState<Settled<T> | null>(null)
  const load = useEffectEvent(loader)

  useEffect(() => {
    let active = true
    load()
      .then((data) => {
        if (active) setSettled({ key, data, error: null })
      })
      .catch((err: unknown) => {
        if (active) setSettled({ key, data: null, error: err instanceof Error ? err.message : String(err) })
      })
    return () => {
      active = false
    }
  }, [key])

  const current = settled?.key === key
  if (current) return { data: settled.data, error: settled.error, loading: false }
  return { data: options.keepPrevious ? (settled?.data ?? null) : null, error: null, loading: true }
}

/** Debounce a fast-changing value (e.g. slider or search box). */
export function useDebounced<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs)
    return () => clearTimeout(id)
  }, [value, delayMs])
  return debounced
}

import { describe, expect, it, vi } from 'vitest'

import { api, ApiError, buildQuery } from './client'

describe('buildQuery', () => {
  it('skips empty values and encodes the rest', () => {
    expect(buildQuery({ a: 1, b: '', c: undefined, d: null, e: 'x y', f: false })).toBe('?a=1&e=x+y&f=false')
    expect(buildQuery({})).toBe('')
  })
})

describe('api client', () => {
  it('raises ApiError with the server detail message', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ detail: 'run not found' }), { status: 404 }),
    )
    await expect(api.overview()).rejects.toEqual(new ApiError(404, 'run not found'))
  })

  it('posts what-if parameters as JSON', async () => {
    const spy = vi
      .spyOn(globalThis, 'fetch')
      .mockResolvedValue(new Response(JSON.stringify({ params: {}, portfolio: [], elapsed_ms: 1 }), { status: 200 }))
    await api.whatIf({ service_level: 0.9 })
    const [url, init] = spy.mock.calls[0] ?? []
    expect(url).toBe('/api/simulation/what-if')
    expect(init?.method).toBe('POST')
    expect(init?.body).toBe(JSON.stringify({ service_level: 0.9 }))
  })

  it('encodes series ids in paths', async () => {
    const spy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('[]', { status: 200 }))
    await api.explanation('A B')
    expect(spy.mock.calls[0]?.[0]).toBe('/api/series/A%20B/explanation')
  })
})

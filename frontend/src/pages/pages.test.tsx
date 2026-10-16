import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'

import { AppRoutes } from '../App'
import { mockFetch, PORTFOLIO, RUN, SENSITIVITY } from '../test/fixtures'

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRoutes />
    </MemoryRouter>,
  )
}

describe('Overview page', () => {
  it('shows headline KPIs from the API', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      mockFetch({
        '/api/overview': RUN,
        '/api/simulation/portfolio': PORTFOLIO,
        '/api/simulation/sensitivity': SENSITIVITY,
      }),
    )
    renderAt('/')
    expect(await screen.findByText('42.5%')).toBeInTheDocument()
    expect(screen.getByText('6.2%')).toBeInTheDocument()
    expect(screen.getByText('Policy scorecard')).toBeInTheDocument()
    expect(screen.getByText(/lead time 7 days/)).toBeInTheDocument()
  })

  it('renders an error state when the API fails', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(mockFetch({}))
    renderAt('/')
    expect(await screen.findByRole('alert')).toHaveTextContent('Something went wrong')
  })
})

describe('What-if page', () => {
  it('re-simulates with the changed parameter when a slider moves', async () => {
    const fetchMock = vi.fn(
      mockFetch({
        '/api/overview': RUN,
        '/api/simulation/what-if': { params: RUN.inventory, portfolio: PORTFOLIO, elapsed_ms: 12 },
      }),
    )
    vi.spyOn(globalThis, 'fetch').mockImplementation(fetchMock)
    renderAt('/what-if')

    const slider = await screen.findByLabelText('Supplier lead time')
    await waitFor(() => expect(screen.getByText('40.0%')).toBeInTheDocument())
    fireEvent.change(slider, { target: { value: '9' } })

    await waitFor(() => {
      const bodies = fetchMock.mock.calls
        .filter(([url]) => String(url).includes('what-if'))
        .map(([, init]) => String(init?.body))
      expect(bodies.at(-1)).toContain('"lead_time_days":9')
    })
  })
})

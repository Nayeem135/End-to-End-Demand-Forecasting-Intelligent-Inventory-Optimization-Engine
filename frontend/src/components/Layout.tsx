import { Suspense } from 'react'
import { NavLink, Outlet } from 'react-router-dom'

import { api } from '../api/client'
import { useApi } from '../hooks/useApi'
import { formatDate } from '../lib/format'
import { Loading } from './ui'

const NAV = [
  { to: '/', label: 'Overview', icon: '◎' },
  { to: '/forecasts', label: 'Forecasts', icon: '↗' },
  { to: '/inventory', label: 'Inventory', icon: '▤' },
  { to: '/what-if', label: 'What-if', icon: '⚙' },
  { to: '/models', label: 'Models', icon: '★' },
]

export function Layout() {
  const run = useApi(() => api.overview(), [])
  return (
    <div className="flex min-h-full">
      <aside className="hidden w-60 shrink-0 flex-col bg-slate-900 text-slate-300 md:flex">
        <div className="px-5 py-6">
          <p className="text-lg font-semibold text-white">Demand&nbsp;Engine</p>
          <p className="mt-1 text-xs text-slate-400">Forecasting &amp; inventory optimisation</p>
        </div>
        <nav className="flex flex-col gap-1 px-3">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition ${
                  isActive ? 'bg-slate-800 text-white' : 'hover:bg-slate-800/60 hover:text-white'
                }`
              }
            >
              <span aria-hidden className="w-4 text-center">
                {item.icon}
              </span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        {run.data && (
          <div className="mt-auto space-y-1 border-t border-slate-800 px-5 py-4 text-xs text-slate-400">
            <p>
              M5 · {run.data.state} · {run.data.n_items} items × {run.data.n_stores} stores
            </p>
            <p>
              Test period from {formatDate(run.data.test_start)} {run.data.test_start.slice(0, 4)}
            </p>
            <p className="truncate">Run {run.data.run_id}</p>
          </div>
        )}
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <nav className="flex gap-2 overflow-x-auto border-b border-slate-200 bg-white px-4 py-2 md:hidden">
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.to === '/'} className="whitespace-nowrap px-2 py-1 text-sm">
              {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 md:px-8">
          <Suspense fallback={<Loading />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  )
}

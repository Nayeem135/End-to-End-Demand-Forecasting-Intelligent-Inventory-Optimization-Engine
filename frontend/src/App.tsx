import { lazy } from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'

import { Layout } from './components/Layout'

// Pages are code-split so the first screen loads only what it needs.
const OverviewPage = lazy(() => import('./pages/OverviewPage').then((m) => ({ default: m.OverviewPage })))
const ForecastsPage = lazy(() => import('./pages/ForecastsPage').then((m) => ({ default: m.ForecastsPage })))
const InventoryPage = lazy(() => import('./pages/InventoryPage').then((m) => ({ default: m.InventoryPage })))
const WhatIfPage = lazy(() => import('./pages/WhatIfPage').then((m) => ({ default: m.WhatIfPage })))
const ModelsPage = lazy(() => import('./pages/ModelsPage').then((m) => ({ default: m.ModelsPage })))

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<OverviewPage />} />
        <Route path="forecasts" element={<ForecastsPage />} />
        <Route path="inventory" element={<InventoryPage />} />
        <Route path="what-if" element={<WhatIfPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  )
}

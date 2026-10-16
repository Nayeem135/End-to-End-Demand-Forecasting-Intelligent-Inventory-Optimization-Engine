import { useState } from 'react'

import { api } from '../api/client'
import { HorizontalBarChart, LeaderboardChart } from '../components/charts'
import { PageHeader } from '../components/PageHeader'
import { Badge, Card, ErrorState, Loading } from '../components/ui'
import { useApi } from '../hooks/useApi'
import { featureLabel, formatNumber, formatPercent, modelLabel } from '../lib/format'

export function ModelsPage() {
  const [split, setSplit] = useState<'validation' | 'test'>('test')
  const board = useApi(() => api.leaderboard(split), [split])
  const shap = useApi(() => api.globalExplanation(15), [])

  return (
    <>
      <PageHeader
        title="Model comparison"
        description="Seven forecasters evaluated with rolling-origin backtesting. The champion is chosen on validation folds only; the test split is reported for honesty."
      >
        <div className="flex rounded-lg border border-slate-200 bg-white p-0.5 text-sm">
          {(['validation', 'test'] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setSplit(s)}
              className={`rounded-md px-3 py-1 capitalize ${split === s ? 'bg-slate-900 text-white' : 'text-slate-600'}`}
            >
              {s}
            </button>
          ))}
        </div>
      </PageHeader>
      {board.loading && <Loading />}
      {board.error && <ErrorState message={board.error} />}
      {board.data && (
        <div className="grid gap-6 lg:grid-cols-5">
          <Card className="lg:col-span-2" title="WRMSSE by model" subtitle="Revenue-weighted, scale-free error (M5 metric)">
            <LeaderboardChart scores={board.data} />
          </Card>
          <Card className="overflow-x-auto lg:col-span-3" title="Leaderboard">
            <table className="w-full min-w-[560px] text-sm">
              <thead className="text-left text-xs uppercase text-slate-500">
                <tr>
                  <th className="pb-2">#</th>
                  <th className="pb-2">Model</th>
                  <th className="pb-2 text-right">WRMSSE</th>
                  <th className="pb-2 text-right">MAE</th>
                  <th className="pb-2 text-right">Bias</th>
                  <th className="pb-2 text-right">80% coverage</th>
                  <th className="pb-2 text-right">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {board.data.map((s) => (
                  <tr key={s.model}>
                    <td className="py-2 text-slate-500">{s.rank}</td>
                    <td className="py-2 font-medium">
                      {modelLabel(s.model)} {s.is_champion && <Badge color="emerald">champion</Badge>}
                    </td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(s.wrmsse, 4)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(s.mae, 3)}</td>
                    <td className="py-2 text-right tabular-nums">{formatNumber(s.bias, 3)}</td>
                    <td className="py-2 text-right tabular-nums">{formatPercent(s.coverage)}</td>
                    <td className="py-2 text-right tabular-nums">{s.seconds.toFixed(0)}s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
      <Card
        className="mt-6"
        title="What drives demand? (global SHAP importance)"
        subtitle="Mean |SHAP| across all forecasts in the final test window — how much each feature moves the LightGBM forecast"
      >
        {shap.loading && <Loading />}
        {shap.error && <ErrorState message={shap.error} />}
        {shap.data && (
          <HorizontalBarChart items={shap.data.map((d) => ({ label: featureLabel(d.feature), value: d.mean_abs_shap }))} />
        )}
      </Card>
    </>
  )
}

import { useMemo } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

interface Props {
  data: { y_true: number; y_prob?: number }[]
}

const BINS = 10

export default function ScoreDistribution({ data }: Props) {
  const chartData = useMemo(() => {
    const hasScores = data.some(d => d.y_prob !== undefined)
    if (!hasScores || data.length === 0) return []

    const fitData = data.filter(d => d.y_true === 1 && d.y_prob !== undefined).map(d => d.y_prob!)
    const noFitData = data.filter(d => d.y_true === 0 && d.y_prob !== undefined).map(d => d.y_prob!)

    const binSize = 1 / BINS
    const bins: { range: string; Fit: number; 'No Fit': number }[] = []

    for (let i = 0; i < BINS; i++) {
      const lo = i * binSize
      const hi = (i + 1) * binSize
      const label = `${(lo * 100).toFixed(0)}-${(hi * 100).toFixed(0)}%`
      bins.push({
        range: label,
        Fit: fitData.filter(s => s >= lo && s < hi).length,
        'No Fit': noFitData.filter(s => s >= lo && s < hi).length,
      })
    }
    return bins
  }, [data])

  if (chartData.length === 0) return null

  return (
    <div>
      <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Distribuição dos Scores por Classe</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={chartData} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="range" tick={{ fill: 'var(--chart-text)', fontSize: 11 }} />
          <YAxis tick={{ fill: 'var(--chart-text)' }} />
          <Tooltip
            contentStyle={{ backgroundColor: 'var(--tooltip-bg)', borderColor: 'var(--chart-grid)', color: 'var(--tooltip-text)' }}
          />
          <Legend wrapperStyle={{ color: 'var(--chart-text)' }} />
          <Bar dataKey="Fit" fill="#22c55e" radius={[2, 2, 0, 0]} />
          <Bar dataKey="No Fit" fill="#ef4444" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

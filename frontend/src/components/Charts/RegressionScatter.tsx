import { useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'

interface Props {
  data: { y_true: number; y_pred: number }[]
}

export default function RegressionScatter({ data }: Props) {
  const { minVal, maxVal } = useMemo(() => {
    let mn = Infinity, mx = -Infinity
    for (const d of data) {
      if (d.y_true < mn) mn = d.y_true
      if (d.y_pred < mn) mn = d.y_pred
      if (d.y_true > mx) mx = d.y_true
      if (d.y_pred > mx) mx = d.y_pred
    }
    return { minVal: mn, maxVal: mx }
  }, [data])

  return (
    <div>
      <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Predito vs Real — Regressão Salarial</h3>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ left: 20, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis
            dataKey="y_true"
            name="Real"
            tick={{ fill: 'var(--chart-text)' }}
            label={{ value: 'Salário Real ($)', position: 'bottom', fill: 'var(--chart-text)' }}
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          />
          <YAxis
            dataKey="y_pred"
            name="Predito"
            tick={{ fill: 'var(--chart-text)' }}
            label={{ value: 'Salário Predito ($)', angle: -90, position: 'insideLeft', fill: 'var(--chart-text)', style: { textAnchor: 'middle' } }}
            domain={['auto', 'auto']}
            tickFormatter={(v: number) => `$${(v / 1000).toFixed(0)}k`}
          />
          <Tooltip
            formatter={(value: number, name: string) => [`$${value.toLocaleString()}`, name === 'y_true' ? 'Real' : 'Predito']}
            contentStyle={{ backgroundColor: 'var(--tooltip-bg)', borderColor: 'var(--chart-grid)', color: 'var(--tooltip-text)' }}
          />
          <ReferenceLine stroke="#ef4444" strokeDasharray="6 3" segment={[{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }]} />
          <Scatter data={data} fill="var(--chart-bar)" opacity={0.6} />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  )
}

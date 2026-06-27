import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts'

interface Props {
  matrix: number[][]
}

export default function ConfusionMatrix({ matrix }: Props) {
  const data = useMemo(() => {
    const labels = ['No Fit', 'Fit']
    const rows: { real: string; pred: string; count: number }[] = []
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        rows.push({ real: labels[i], pred: labels[j], count: matrix[i][j] })
      }
    }
    return rows
  }, [matrix])

  return (
    <div>
      <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Matriz de Confusão</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} layout="vertical" margin={{ left: 60, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis type="number" tick={{ fill: 'var(--chart-text)' }} />
          <YAxis dataKey="real" type="category" tick={{ fill: 'var(--chart-text)' }} />
          <Tooltip
            formatter={(value: number) => value.toLocaleString()}
            contentStyle={{ backgroundColor: 'var(--tooltip-bg)', borderColor: 'var(--chart-grid)', color: 'var(--tooltip-text)' }}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((_, i) => <Cell key={i} fill={`var(--chart-color-${i})`} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

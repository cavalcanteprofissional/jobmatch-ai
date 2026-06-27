import { useMemo } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
} from 'recharts'

interface Props {
  matrix: number[][]
}

const COLORS = ['#dbeafe', '#93c5fd', '#3b82f6', '#1d4ed8']

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
      <h3 className="font-semibold mb-2">Matriz de Confusão</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} layout="vertical" margin={{ left: 60, right: 20, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="real" type="category" />
          <Tooltip formatter={(value: number) => value.toLocaleString()} />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

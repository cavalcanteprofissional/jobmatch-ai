import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface Props {
  scores: number[]
  label: string
  higherBetter: boolean
}

export default function NestedCVChart({ scores, label, higherBetter }: Props) {
  const data = scores.map((s, i) => ({ fold: `Fold ${i + 1}`, score: s }))
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length

  return (
    <div>
      <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Nested CV — {label}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="fold" tick={{ fill: 'var(--chart-text)' }} />
          <YAxis domain={['auto', 'auto']} tick={{ fill: 'var(--chart-text)' }} />
          <Tooltip contentStyle={{ backgroundColor: 'var(--tooltip-bg)', borderColor: 'var(--chart-grid)', color: 'var(--tooltip-text)' }} />
          <ReferenceLine y={mean} stroke="#3b82f6" strokeDasharray="6 3" label={{ value: `Média: ${mean.toFixed(4)}`, fill: 'var(--chart-text)' }} />
          <Bar dataKey="score" fill={higherBetter ? '#22c55e' : '#ef4444'} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

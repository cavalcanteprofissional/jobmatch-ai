import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'

interface Props {
  accuracy: number
  f1_score: number
  precision: number
  recall: number
}

export default function MetricsBar({ accuracy, f1_score, precision, recall }: Props) {
  const data = [
    { name: 'Acurácia', value: accuracy },
    { name: 'F1-Score', value: f1_score },
    { name: 'Precisão', value: precision },
    { name: 'Recall', value: recall },
  ]

  return (
    <div>
      <h3 className="font-semibold mb-2 text-gray-900 dark:text-gray-100">Métricas de Classificação</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
          <XAxis dataKey="name" tick={{ fill: 'var(--chart-text)' }} />
          <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fill: 'var(--chart-text)' }} />
          <Tooltip
            formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
            contentStyle={{ backgroundColor: 'var(--tooltip-bg)', borderColor: 'var(--chart-grid)', color: 'var(--tooltip-text)' }}
          />
          <ReferenceLine y={0.7} stroke="#ef4444" strokeDasharray="6 3" label={{ value: 'Meta 70%', fill: 'var(--chart-text)' }} />
          <Bar dataKey="value" fill="var(--chart-bar)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

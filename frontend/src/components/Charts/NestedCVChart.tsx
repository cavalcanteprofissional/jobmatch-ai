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
      <h3 className="font-semibold mb-2">Nested CV — {label}</h3>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="fold" />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip />
          <ReferenceLine y={mean} stroke="blue" strokeDasharray="6 3" label={`Média: ${mean.toFixed(4)}`} />
          <Bar dataKey="score" fill={higherBetter ? '#2ca02c' : '#d62728'} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

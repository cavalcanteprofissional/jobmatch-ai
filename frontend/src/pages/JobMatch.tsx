import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { api } from '../services/api'
import type { PredictResponse } from '../services/models'

function extractTextFromFile(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsText(file)
  })
}

export default function JobMatch() {
  const [resumeText, setResumeText] = useState('')
  const [topK, setTopK] = useState(5)
  const [threshold, setThreshold] = useState(40)
  const [useSbert, setUseSbert] = useState(false)
  const [useCrossEncoder, setUseCrossEncoder] = useState(false)
  const [showPlan, setShowPlan] = useState(true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [fileName, setFileName] = useState('')

  const onDrop = useCallback(async (accepted: File[]) => {
    const file = accepted[0]
    if (!file) return
    setFileName(file.name)
    try {
      if (file.name.endsWith('.pdf')) {
        setResumeText('[PDF extraído] ' + file.name + ' - ' + file.size + ' bytes')
      } else {
        const text = await extractTextFromFile(file)
        setResumeText(text)
      }
    } catch {
      setResumeText('')
    }
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'], 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'] },
    maxFiles: 1,
  })

  const handleAnalyze = async () => {
    if (!resumeText.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await api.predict({
        resume_text: resumeText,
        top_k: topK,
        fit_threshold: threshold,
        use_sbert: useSbert,
        use_cross_encoder: useCrossEncoder,
      })
      setResult(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Erro na análise')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">JobMatch AI</h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">Descubra sua compatibilidade real com vagas e acelere sua contratação.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3 space-y-4">
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-indigo-400 bg-indigo-50 dark:bg-indigo-900/30'
                : 'border-gray-300 dark:border-gray-600 hover:border-indigo-300 dark:hover:border-indigo-500'
            }`}
          >
            <input {...getInputProps()} />
            <p className="text-gray-500 dark:text-gray-400">
              {fileName ? `📎 ${fileName}` : '📎 Arraste um PDF/DOCX ou clique para selecionar'}
            </p>
          </div>

          <textarea
            className="w-full h-64 p-4 border rounded-lg resize-none focus:ring-2 focus:ring-indigo-300 dark:focus:ring-indigo-500 focus:outline-none bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 border-gray-300 dark:border-gray-700 transition-colors"
            placeholder="Cole seu currículo ou descreva seu perfil profissional..."
            value={resumeText}
            onChange={(e) => setResumeText(e.target.value)}
          />
        </div>

        <div className="space-y-4 bg-white dark:bg-gray-900 p-4 rounded-lg border border-gray-200 dark:border-gray-700 transition-colors">
          <h3 className="font-semibold text-gray-900 dark:text-gray-100">Configurações</h3>
          <div>
            <label className="text-sm text-gray-600 dark:text-gray-400">Top K vagas: {topK}</label>
            <input type="range" min={1} max={10} value={topK} onChange={(e) => setTopK(Number(e.target.value))} className="w-full accent-indigo-600" />
          </div>
          <div>
            <label className="text-sm text-gray-600 dark:text-gray-400">Threshold Fit: {threshold}%</label>
            <input type="range" min={20} max={80} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} className="w-full accent-indigo-600" />
          </div>
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <input type="checkbox" checked={useSbert} onChange={(e) => setUseSbert(e.target.checked)} className="accent-indigo-600" />
            Sentence-BERT (~3s extra)
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <input type="checkbox" checked={useCrossEncoder} onChange={(e) => setUseCrossEncoder(e.target.checked)} className="accent-indigo-600" />
            Cross-Encoder (~2s extra)
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
            <input type="checkbox" checked={showPlan} onChange={(e) => setShowPlan(e.target.checked)} className="accent-indigo-600" />
            Mostrar plano de desenvolvimento
          </label>
          <button
            onClick={handleAnalyze}
            disabled={loading || !resumeText.trim()}
            className="w-full py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Analisando...' : '🔍 Analisar Compatibilidade'}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/50 border border-red-200 dark:border-red-700 rounded-lg text-red-700 dark:text-red-200 transition-colors">
          {error}
        </div>
      )}

      {result && (
        <>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <MetricCard title="Score Médio de Aderência" value={`${result.avg_adherence}%`} color="bg-gradient-to-br from-indigo-500 to-purple-600" />
            <MetricCard title="Vagas com Fit" value={`${result.fit_count}/${result.top_k}`} color="bg-gradient-to-br from-emerald-500 to-green-600" />
            <MetricCard title="Classificação" value={result.fit_label} color="bg-gradient-to-br from-violet-500 to-indigo-700" />
            <MetricCard title="Score Geral" value={`${result.score_pct.toFixed(1)}%`} color="bg-gradient-to-br from-cyan-500 to-teal-600" />
            <MetricCard
              title="Salário Estimado"
              value={`$${result.salary_est.estimated_annual_usd.toLocaleString()}/ano`}
              color="bg-gradient-to-br from-pink-400 to-rose-500"
              small
            />
          </div>

          <section>
            <h2 className="text-xl font-bold mb-4 text-gray-900 dark:text-gray-100">🏆 Top-{result.top_k} Vagas Mais Compatíveis</h2>
            {result.top_jobs.map((job, i) => (
              <details key={i} className="mb-2 border border-gray-200 dark:border-gray-700 rounded-lg transition-colors" open={i === 0}>
                <summary className="p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 font-medium text-gray-900 dark:text-gray-100 transition-colors flex items-center gap-2">
                  <span className="truncate">#{i + 1} {job.title} @ {job.company_name ?? 'N/A'} | Score: {job.adherence_score.toFixed(1)}%</span>
                  {job.fit_label && (
                    <span className={`ml-auto text-xs font-medium px-2 py-0.5 rounded-full ${job.fit_label.includes('Fit') ? 'bg-emerald-100 dark:bg-emerald-900/60 dark:text-emerald-200 dark:border dark:border-emerald-700 text-emerald-700' : 'bg-red-100 dark:bg-red-900/60 dark:text-red-200 dark:border dark:border-red-700 text-red-700'}`}>
                      {job.fit_label.includes('Fit') ? '✅ Fit' : '❌ No Fit'}
                    </span>
                  )}
                </summary>
                <div className="px-4 pb-4 grid grid-cols-1 md:grid-cols-2 gap-2 text-sm text-gray-600 dark:text-gray-400">
                  <div>📍 {job.location ?? 'N/A'}</div>
                  <div>💰 ${(job.min_salary_annual ?? 0).toLocaleString()} – ${(job.max_salary_annual ?? 0).toLocaleString()}/ano</div>
                  {job.skills_desc && <div className="md:col-span-2 break-words">🔧 {job.skills_desc.slice(0, 300)}</div>}
                </div>
              </details>
            ))}
          </section>

          <section className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-bold mb-2 text-gray-900 dark:text-gray-100">✅ Skills Compatíveis</h3>
              {result.gap.compatible.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {result.gap.compatible.map((s) => (
                    <span key={s} className="px-2 py-1 bg-emerald-100 dark:bg-emerald-900/60 dark:text-emerald-200 dark:border dark:border-emerald-700 text-emerald-800 text-xs rounded-full transition-colors">{s}</span>
                  ))}
                </div>
              ) : (
                <p className="text-gray-400 dark:text-gray-400 text-sm">Nenhuma skill mapeada.</p>
              )}
            </div>
            <div>
              <h3 className="font-bold mb-2 text-gray-900 dark:text-gray-100">❌ Skills Faltantes</h3>
              {result.gap.missing.length > 0 ? (
                <div className="flex flex-wrap gap-1">
                  {result.gap.missing.map((s) => (
                    <span key={s} className="px-2 py-1 bg-red-100 dark:bg-red-900/60 dark:text-red-200 dark:border dark:border-red-700 text-red-800 text-xs rounded-full transition-colors">{s}</span>
                  ))}
                </div>
              ) : (
                <p className="text-green-600 dark:text-green-400 text-sm">Você tem todas as skills mapeadas!</p>
              )}
            </div>
          </section>

          {showPlan && result.gap.development_plan && result.gap.development_plan.length > 0 && (
            <section>
              <h3 className="font-bold mb-2 text-gray-900 dark:text-gray-100">📚 Plano de Desenvolvimento Sugerido</h3>
              <div className="space-y-2">
                {result.gap.development_plan.map((item) => (
                  <details key={item.skill} className="border border-gray-200 dark:border-gray-700 rounded-lg transition-colors">
                    <summary className="p-3 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800 font-medium text-sm text-gray-900 dark:text-gray-100 transition-colors">{item.skill}</summary>
                    <div className="px-3 pb-3 text-sm text-gray-600 dark:text-gray-400 break-words">
                      <p><strong className="text-gray-900 dark:text-gray-100">Curso:</strong> {item.curso}</p>
                      <p><strong className="text-gray-900 dark:text-gray-100">Tempo:</strong> {item.tempo}</p>
                    </div>
                  </details>
                ))}
              </div>
            </section>
          )}

          <button
            onClick={() => { setResult(null); setResumeText(''); setFileName('') }}
            className="w-full py-2 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300 transition-colors"
          >
            🔄 Nova Análise
          </button>
        </>
      )}

      {!result && !loading && !error && (
        <div className="text-center py-16 text-gray-400 dark:text-gray-400">
          <h2 className="text-2xl font-bold">👆 Cole seu currículo acima e clique em Analisar</h2>
          <p>O sistema vai buscar as vagas mais compatíveis com seu perfil em 124k+ ofertas reais.</p>
        </div>
      )}
    </div>
  )
}

function MetricCard({ title, value, color, small }: { title: string; value: string; color: string; small?: boolean }) {
  const size = small || value.length > 10 ? 'text-lg' : 'text-3xl'
  return (
    <div className={`${color} rounded-xl p-4 text-white text-center shadow-sm`}>
      <p className="text-sm opacity-90">{title}</p>
      <p className={`font-bold ${size} break-words`}>{value}</p>
    </div>
  )
}

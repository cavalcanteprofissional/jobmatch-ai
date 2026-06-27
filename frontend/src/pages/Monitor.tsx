import { useEffect, useState } from 'react'
import { api } from '../services/api'
import type { ModelMetrics, ApiMetrics, EvalClassificationResponse, EvalRegressionResponse } from '../services/models'
import ConfusionMatrix from '../components/Charts/ConfusionMatrix'
import MetricsBar from '../components/Charts/MetricsBar'
import NestedCVChart from '../components/Charts/NestedCVChart'
import RegressionScatter from '../components/Charts/RegressionScatter'
import ScoreDistribution from '../components/Charts/ScoreDistribution'

export default function Monitor() {
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null)
  const [apiMetrics, setApiMetrics] = useState<ApiMetrics | null>(null)
  const [evalClf, setEvalClf] = useState<EvalClassificationResponse | null>(null)
  const [evalReg, setEvalReg] = useState<EvalRegressionResponse | null>(null)
  const [tab, setTab] = useState<'ml' | 'api'>('ml')

  useEffect(() => {
    api.modelMetrics().then(setModelMetrics).catch(() => setModelMetrics(null))
    api.metrics().then(setApiMetrics).catch(() => setApiMetrics(null))
    api.evalClassification().then(setEvalClf).catch(() => setEvalClf(null))
    api.evalRegression().then(setEvalReg).catch(() => setEvalReg(null))
  }, [])

  const clf = modelMetrics?.classification
  const reg = modelMetrics?.regression
  const info = modelMetrics?.model_info

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">📊 JobMatch AI — Monitor</h1>

      <div className="flex gap-2">
        <button
          onClick={() => setTab('ml')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab === 'ml'
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
          }`}
        >
          📊 Modelo ML
        </button>
        <button
          onClick={() => setTab('api')}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            tab === 'api'
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
          }`}
        >
          ⚡ API
        </button>
      </div>

      {tab === 'ml' && modelMetrics && clf && reg && info ? (
        <div className="space-y-6">
          <section>
            <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">🔍 Classificação Fit x No Fit</h2>
            <div className="grid grid-cols-3 sm:grid-cols-6 gap-3">
              <Metric label="Modelo" value={clf.model_type} />
              <Metric label="Acurácia" value={`${(clf.accuracy * 100).toFixed(1)}%`} />
              <Metric label="F1-Score" value={`${(clf.f1_score * 100).toFixed(1)}%`} />
              <Metric label="Precisão" value={`${(clf.precision * 100).toFixed(1)}%`} />
              <Metric label="Recall" value={`${(clf.recall * 100).toFixed(1)}%`} />
              <Metric label="Amostras" value={clf.test_samples.toLocaleString()} />
              {clf.best_candidate && <Metric label="Melhor Candidato" value={clf.best_candidate} className="truncate max-w-[140px]" />}
              {clf.best_params && (
                <details className="col-span-full border border-gray-200 dark:border-gray-700 rounded-lg p-2 text-xs font-mono text-gray-700 dark:text-gray-300">
                  <summary className="cursor-pointer font-medium">Melhores Params</summary>
                  {Object.entries(clf.best_params).map(([k, v]) => (
                    <div key={k} className="flex gap-2 ml-2">
                      <span className="font-semibold whitespace-nowrap">{k}:</span>
                      <span className="break-all">{String(v)}</span>
                    </div>
                  ))}
                </details>
              )}
            </div>
          </section>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {clf.confusion_matrix && <ConfusionMatrix matrix={clf.confusion_matrix} />}
            <MetricsBar accuracy={clf.accuracy} f1_score={clf.f1_score} precision={clf.precision} recall={clf.recall} />
          </div>

          {clf.nested_cv && (
            <section>
              <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">🎯 Nested CV — Classificação</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <NestedCVChart scores={clf.nested_cv.scores} label="F1 Score" higherBetter />
                <div className="grid grid-cols-2 gap-3">
                  <Metric label="F1 Médio" value={`${(clf.nested_cv.mean * 100).toFixed(1)}%`} />
                  <Metric label="Desvio Padrão" value={`±${(clf.nested_cv.std * 100).toFixed(2)}%`} />
                </div>
              </div>
            </section>
          )}

          {clf.sbert && (
            <section>
              <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">🧠 TF-IDF vs Sentence-BERT</h2>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Metric label="SBERT Modelo" value={clf.sbert.model_type} />
                <Metric label="SBERT F1" value={`${(clf.sbert.f1_score * 100).toFixed(1)}%`} />
              </div>
            </section>
          )}

          {evalClf && evalClf.data.length > 0 && (
            <section>
              <ScoreDistribution data={evalClf.data} />
            </section>
          )}

          <section>
            <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">💰 Regressão Salarial</h2>
            <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
              <Metric label="Modelo" value={reg.model_type} />
              <Metric label="RMSE" value={`$${reg.rmse.toLocaleString()}`} />
              <Metric label="MAE" value={`$${reg.mae.toLocaleString()}`} />
              <Metric label="R²" value={`${(reg.r2 * 100).toFixed(1)}%`} />
              <Metric label="Amostras" value={reg.test_samples.toLocaleString()} />
              {reg.best_candidate && <Metric label="Melhor Candidato" value={reg.best_candidate} className="truncate max-w-[140px]" />}
              {reg.best_params && (
                <details className="col-span-full border border-gray-200 dark:border-gray-700 rounded-lg p-2 text-xs font-mono text-gray-700 dark:text-gray-300">
                  <summary className="cursor-pointer font-medium">Melhores Params</summary>
                  {Object.entries(reg.best_params).map(([k, v]) => (
                    <div key={k} className="flex gap-2 ml-2">
                      <span className="font-semibold whitespace-nowrap">{k}:</span>
                      <span className="break-all">{String(v)}</span>
                    </div>
                  ))}
                </details>
              )}
            </div>
          </section>

          {reg.nested_cv && (
            <section>
              <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">🎯 Nested CV — Regressão</h2>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <NestedCVChart scores={reg.nested_cv.scores} label="RMSE ($)" higherBetter={false} />
                <div className="grid grid-cols-2 gap-3">
                  <Metric label="RMSE Médio" value={`$${reg.nested_cv.mean.toLocaleString()}`} />
                  <Metric label="Desvio Padrão" value={`±$${reg.nested_cv.std.toLocaleString()}`} />
                </div>
              </div>
            </section>
          )}

          {evalReg && evalReg.data.length > 0 && (
            <section>
              <RegressionScatter data={evalReg.data} />
            </section>
          )}

          <section>
            <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">⚙️ Informações do Modelo</h2>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <Metric label="Features TF-IDF" value={typeof info.vectorizer_features === 'number' ? info.vectorizer_features.toLocaleString() : 'N/A'} />
              <Metric label="Total de Vagas" value={info.total_jobs.toLocaleString()} />
              <Metric label="Vagas c/ Salário" value={info.jobs_with_salary.toLocaleString()} />
              <Metric label="Pares de Treino" value={info.training_pairs.toLocaleString()} />
            </div>
          </section>
        </div>
      ) : tab === 'ml' && !modelMetrics ? (
        <div className="p-6 bg-yellow-50 dark:bg-yellow-900/50 border border-yellow-200 dark:border-yellow-700 rounded-lg text-yellow-800 dark:text-yellow-200 transition-colors">
          Métricas dos modelos não disponíveis. Execute o treino dos modelos (reload_eval.py) e reinicie a API.
        </div>
      ) : null}

      {tab === 'api' && apiMetrics ? (
        <div className="space-y-6">
          <div className="grid grid-cols-4 gap-3">
            <Metric label="Requisições" value={apiMetrics.total_requests.toLocaleString()} />
            <Metric label="Erros" value={apiMetrics.total_errors.toLocaleString()} />
            <Metric label="Taxa de Erro" value={`${apiMetrics.error_rate_pct}%`} />
            <Metric label="Uptime" value={`${Math.floor(apiMetrics.uptime_seconds / 3600)}h ${Math.floor((apiMetrics.uptime_seconds % 3600) / 60)}m`} />
          </div>

          {Object.keys(apiMetrics.endpoints).length > 0 && (
            <section>
              <h2 className="text-lg font-bold mb-3 text-gray-900 dark:text-gray-100">📈 Métricas por Endpoint</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border-collapse">
                  <thead>
                    <tr className="bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
                      <th className="p-2 text-left">Endpoint</th>
                      <th className="p-2 text-left">Req</th>
                      <th className="p-2 text-left">Erros</th>
                      <th className="p-2 text-left">Lat Média</th>
                      <th className="p-2 text-left">P99</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(apiMetrics.endpoints).map(([ep, m]) => (
                      <tr key={ep} className="border-b border-gray-200 dark:border-gray-700 text-gray-700 dark:text-gray-300">
                        <td className="p-2 font-mono text-xs max-w-[200px] truncate" title={ep}>{ep}</td>
                        <td className="p-2">{m.requests}</td>
                        <td className="p-2">{m.errors}</td>
                        <td className="p-2">{m.latency_ms_avg.toFixed(1)}ms</td>
                        <td className="p-2">{m.latency_ms_p99.toFixed(1)}ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      ) : tab === 'api' && !apiMetrics ? (
        <div className="p-6 bg-yellow-50 dark:bg-yellow-900/50 border border-yellow-200 dark:border-yellow-700 rounded-lg text-yellow-800 dark:text-yellow-200 transition-colors">
          API não disponível. Execute a API primeiro.
        </div>
      ) : null}
    </div>
  )
}

function Metric({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg p-3 text-center transition-colors">
      <p className="text-xs text-gray-500 dark:text-gray-400">{label}</p>
      <p className={`text-lg font-bold text-gray-900 dark:text-gray-100 ${className ?? ''}`}>{value}</p>
    </div>
  )
}

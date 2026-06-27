import type {
  PredictRequest, PredictResponse, ModelInfo, ApiMetrics,
  ModelMetrics, EvalClassificationResponse, EvalRegressionResponse,
} from './models'

const CLOUD_API = import.meta.env.VITE_API_URL || ''
const LOCAL_API = 'http://localhost:8000'
const DEV_API = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const urls = CLOUD_API
    ? [`${CLOUD_API}${path}`, `${LOCAL_API}${path}`]
    : [`${DEV_API}${path}`]

  let lastError: Error | null = null
  for (const url of urls) {
    try {
      const res = await fetch(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      })
      if (!res.ok) {
        const body = await res.text()
        throw new Error(`${res.status}: ${body}`)
      }
      return res.json()
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e))
    }
  }
  throw lastError!
}

export const api = {
  predict(data: PredictRequest): Promise<PredictResponse> {
    return request<PredictResponse>('/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },

  health(): Promise<{ status: string; models_loaded: boolean; jobs_count: number }> {
    return request('/health')
  },

  modelInfo(): Promise<ModelInfo> {
    return request<ModelInfo>('/models/info')
  },

  modelMetrics(): Promise<ModelMetrics> {
    return request<ModelMetrics>('/models/metrics')
  },

  metrics(): Promise<ApiMetrics> {
    return request<ApiMetrics>('/metrics')
  },

  evalClassification(): Promise<EvalClassificationResponse> {
    return request<EvalClassificationResponse>('/eval/classification')
  },

  evalRegression(): Promise<EvalRegressionResponse> {
    return request<EvalRegressionResponse>('/eval/regression')
  },
}

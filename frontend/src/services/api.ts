import type {
  PredictRequest, PredictResponse, ModelInfo, ApiMetrics,
  ModelMetrics, EvalClassificationResponse, EvalRegressionResponse,
} from './models'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status}: ${body}`)
  }
  return res.json()
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

import type {
  PredictRequest, PredictResponse, ModelInfo, ApiMetrics,
} from './models'

const API_BASE = '/api'

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

  metrics(): Promise<ApiMetrics> {
    return request<ApiMetrics>('/metrics')
  },
}

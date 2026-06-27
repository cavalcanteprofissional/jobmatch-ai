import type {
  PredictRequest, PredictResponse, ModelInfo, ApiMetrics,
  ModelMetrics, EvalClassificationResponse, EvalRegressionResponse,
} from './models'

const CLOUD_API = import.meta.env.VITE_API_URL || ''
const DEV_API = '/api'

const TIMEOUT_MS = 15_000
const MAX_RETRIES = 3
const BACKOFF_MS = [2_000, 4_000, 8_000]

function fetchWithTimeout(url: string, options: RequestInit, timeoutMs: number): Promise<Response> {
  const controller = new AbortController()
  const id = setTimeout(() => controller.abort(), timeoutMs)
  return fetch(url, { ...options, signal: controller.signal }).finally(() => clearTimeout(id))
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const url = CLOUD_API ? `${CLOUD_API}${path}` : `${DEV_API}${path}`
  const attempts = CLOUD_API ? MAX_RETRIES : 1

  let lastError: Error | null = null
  for (let i = 0; i < attempts; i++) {
    if (i > 0) {
      await new Promise((r) => setTimeout(r, BACKOFF_MS[i - 1] ?? BACKOFF_MS[BACKOFF_MS.length - 1]))
    }
    try {
      const res = await fetchWithTimeout(url, {
        headers: { 'Content-Type': 'application/json' },
        ...options,
      }, TIMEOUT_MS)
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

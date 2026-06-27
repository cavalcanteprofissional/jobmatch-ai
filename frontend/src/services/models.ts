export interface PredictRequest {
  resume_text: string
  top_k: number
  fit_threshold: number
  use_sbert: boolean
  use_cross_encoder: boolean
}

export interface SalaryEstimate {
  estimated_annual_usd: number
  range_low: number
  range_high: number
}

export interface GapAnalysis {
  compatible: string[]
  missing: string[]
  development_plan?: { skill: string; curso: string; tempo: string }[]
}

export interface JobRow {
  title: string
  company_name?: string
  location?: string
  min_salary_annual?: number
  max_salary_annual?: number
  skills_desc?: string
  adherence_score: number
  fit_label: string
  [key: string]: unknown
}

export interface PredictResponse {
  score_pct: number
  fit_label: string
  avg_adherence: number
  fit_count: number
  top_k: number
  employability_score: number
  salary_est: SalaryEstimate
  gap: GapAnalysis
  top_jobs: JobRow[]
}

export interface ModelInfo {
  vectorizer_features: number
  classifier_type: string
  regressor_type: string
  jobs_count: number
  skills_map_titles: number
}

export interface EndpointMetrics {
  requests: number
  errors: number
  latency_ms_avg: number
  latency_ms_min: number
  latency_ms_max: number
  latency_ms_p99: number
}

export interface ApiMetrics {
  uptime_seconds: number
  total_requests: number
  total_errors: number
  error_rate_pct: number
  endpoints: Record<string, EndpointMetrics>
}

export interface NestedCV {
  scores: number[]
  mean: number
  std: number
}

export interface ClassificationMetrics {
  vectorizer?: string
  model_type: string
  accuracy: number
  f1_score: number
  precision: number
  recall: number
  test_samples: number
  confusion_matrix: number[][]
  nested_cv?: NestedCV
  best_params?: Record<string, unknown>
  best_candidate?: string
  sbert?: {
    model_type: string
    accuracy: number
    f1_score: number
    nested_cv_mean: number
    best_candidate?: string
  }
}

export interface RegressionMetrics {
  model_type: string
  rmse: number
  mae: number
  r2: number
  test_samples: number
  nested_cv?: NestedCV
  best_params?: Record<string, unknown>
  best_candidate?: string
}

export interface ModelMetrics {
  classification: ClassificationMetrics
  regression: RegressionMetrics
  model_info: {
    vectorizer: string
    vectorizer_features?: number
    total_jobs: number
    jobs_with_salary: number
    training_pairs: number
  }
}

export interface EvalRow {
  y_true: number
  y_pred: number
  y_prob?: number
  y_true_label?: string
  y_pred_label?: string
}

export interface EvalClassificationResponse {
  data: EvalRow[]
  total: number
}

export interface EvalRegressionResponse {
  data: EvalRow[]
  total: number
}



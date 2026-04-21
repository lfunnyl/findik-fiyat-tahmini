// dashboard/src/lib/api.ts
// API client — tüm fetch mantığı tek yerde

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

export interface MonthlyPrediction {
  ay: number
  ay_adi: string
  kur: number
  reel_usd: number
  nominal_usd: number
  tl: number
  ci_low: number
  ci_high: number
}

export interface PredictResponse {
  predictions: MonthlyPrediction[]
  q_hat: number
  model_info: {
    algorithm: string
    weights: Record<string, number>
    test_mape: number
    test_r2: number
  }
}

export interface WhatIfResponse {
  whatif_tl: number
  baz_tl: number
  delta_tl: number
  delta_pct: number
  whatif_reel_usd: number
  whatif_nominal_usd: number
}

export interface ModelScores {
  [model: string]: {
    MAE: number
    RMSE: number
    R2: number
    MAPE: number
  }
}

export interface TmoPrediction {
  pred_2026: number
  ci_p25: number
  ci_p75: number
  ci_p5: number
  ci_p95: number
  mape_loo: number
  r2_loo: number
  tmo_2025: number
  buyume_pct: number
}

export interface CausalEffect {
  treatment: string
  outcome: string
  average_treatment_effect: number
  yorum: string
}

export interface HistoricalPrice {
  Tarih: string
  Serbest_Piyasa_TL_kg: number
  Fiyat_USD_kg: number
  Fiyat_RealUSD_kg: number
}

export interface ModelInfo {
  model: string
  version: string
  target: string
  data_rows: number
  data_start: string | null
  data_end: string | null
  features_n: number
  weights: Record<string, number>
  config: {
    top_n_features: number
    cpi_base_year: number
  }
}

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export const api = {
  health: () => apiFetch<{ status: string; model: string; data_rows: number }>('/health'),

  predict: (usd_try: number, aylik_kur_artis: number) =>
    apiFetch<PredictResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ usd_try, aylik_kur_artis }),
    }),

  whatif: (params: {
    usd_try: number
    brent_petrol?: number
    altin_ons?: number
    asgari_ucret?: number
    rekolte_degisim_pct: number
    ihracat_degisim_pct: number
  }) =>
    apiFetch<WhatIfResponse>('/api/whatif', {
      method: 'POST',
      body: JSON.stringify(params),
    }),

  scores: () => apiFetch<ModelScores>('/api/scores'),

  tmo: () => apiFetch<TmoPrediction>('/api/tmo'),

  causal: () => apiFetch<CausalEffect>('/api/causal'),

  history: (months = 36) => apiFetch<HistoricalPrice[]>(`/api/history?months=${months}`),

  info: () => apiFetch<ModelInfo>('/api/info'),
}

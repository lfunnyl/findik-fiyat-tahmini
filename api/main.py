"""
api/main.py
===========
Fındık Fiyat Tahmin Sistemi — FastAPI Backend

Endpoint'ler:
  GET  /health          → Sistem sağlık kontrolü
  GET  /api/info        → Model bilgisi ve istatistikler
  POST /api/predict     → 2026 aylık tahminler
  POST /api/whatif      → What-If senaryo analizi
  GET  /api/scores      → Model performans skorları
  GET  /api/tmo         → TMO 2026 tahmini
  GET  /api/causal      → Double ML nedensel etki
  GET  /api/weather     → Hava durumu & iklim riski

Deploy:
  Render: uvicorn api.main:app --host 0.0.0.0 --port $PORT
  Local:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Dizin Yapısı ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_PATH  = BASE_DIR / "data" / "processed" / "master_features.csv"
CFG_PATH   = BASE_DIR / "config.yaml"

sys.path.insert(0, str(BASE_DIR / "src" / "models"))

# ─── Loglama ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Konfig ───────────────────────────────────────────────────────────────────
with open(CFG_PATH, "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

US_CPI_TABLE   = {int(k): v for k, v in CFG["us_cpi"].items()}
CPI_BAZ_YILI   = int(CFG.get("cpi_base_year", 2024))
TARGET         = CFG.get("target", "Fiyat_RealUSD_kg")
TOP_N_FEATURES = int(CFG["model"].get("top_n_features", 20))
DROP_COLS      = list(CFG.get("drop_cols", []))

AYLAR_TR = {
    1: "Ocak",    2: "Şubat",   3: "Mart",    4: "Nisan",
    5: "Mayıs",   6: "Haziran", 7: "Temmuz",  8: "Ağustos",
    9: "Eylül",  10: "Ekim",   11: "Kasım",  12: "Aralık",
}

# ─── Global Model State ────────────────────────────────────────────────────────
_state: dict = {}


def _load_models() -> None:
    """Uygulama başlangıcında modelleri yükle."""
    logger.info("Modeller yükleniyor...")
    try:
        xgb_bundle   = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        lgb_bundle   = joblib.load(MODELS_DIR / "lightgbm_model.pkl")
        ridge_bundle = joblib.load(MODELS_DIR / "ridge_model.pkl")

        with open(MODELS_DIR / "ensemble_weights.json", "r") as f:
            weights = json.load(f)

        ms_1 = joblib.load(MODELS_DIR / "multistep_1m.pkl")
        ms_3 = joblib.load(MODELS_DIR / "multistep_3m.pkl")
        ms_6 = joblib.load(MODELS_DIR / "multistep_6m.pkl")

        _state["xgb_m"]        = xgb_bundle["model"]
        _state["lgb_m"]        = lgb_bundle["model"]
        _state["ridge_m"]      = ridge_bundle["model"]
        _state["ridge_scaler"] = ridge_bundle["scaler"]
        _state["feat_cols"]    = xgb_bundle["features"]
        _state["weights"]      = weights
        _state["ms_1"]         = ms_1
        _state["ms_3"]         = ms_3
        _state["ms_6"]         = ms_6

        logger.info(
            f"Ensemble ağırlıkları — XGBoost: {weights.get('XGBoost',0):.3f} "
            f"| Ridge: {weights.get('Ridge',0):.3f}"
        )

        # Veri yükle
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        df = df.sort_values("Tarih").reset_index(drop=True)

        # USD lag feature'ları ekle (app.py ile senkron)
        df["USD_Lag1"]     = df["Fiyat_USD_kg"].shift(1)
        df["USD_Lag2"]     = df["Fiyat_USD_kg"].shift(2)
        df["USD_Lag3"]     = df["Fiyat_USD_kg"].shift(3)
        df["USD_Lag12"]    = df["Fiyat_USD_kg"].shift(12)
        df["USD_MoM_pct"]  = df["Fiyat_USD_kg"].pct_change(1) * 100
        df["USD_YoY_pct"]  = df["Fiyat_USD_kg"].pct_change(12) * 100
        df["RealUSD_Lag1"] = df["Fiyat_RealUSD_kg"].shift(1)
        df["RealUSD_Lag3"] = df["Fiyat_RealUSD_kg"].shift(3)
        df = df.bfill().ffill()

        _state["df"] = df

        # Feature kolonlarını belirle
        drop_existing = [c for c in DROP_COLS if c in df.columns]
        X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
        split_idx = int(len(df) * 0.80)
        y_log = np.log1p(df[TARGET])
        corr = X.iloc[:split_idx].corrwith(y_log.iloc[:split_idx]).abs().dropna()
        _state["sel_cols"] = corr.nlargest(TOP_N_FEATURES).index.tolist()
        _state["X_all"]    = X

        logger.info(f"Veri yüklendi: {len(df)} satır, {X.shape[1]} feature → Top-{TOP_N_FEATURES} seçildi")

    except Exception as exc:
        logger.error(f"Model yüklenemedi: {exc}", exc_info=True)
        raise RuntimeError(f"Model yüklenemedi: {exc}") from exc


# ─── Lifespan (startup/shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models()
    yield
    _state.clear()
    logger.info("Uygulama kapatıldı.")


# ─── FastAPI Uygulaması ────────────────────────────────────────────────────────
app = FastAPI(
    title="🌰 Fındık Fiyat Tahmin API",
    description=(
        "Türkiye fındık serbest piyasa fiyatı tahmin sistemi. "
        "Weighted Ensemble (XGBoost + Ridge), Conformal Prediction, "
        "Double ML Causal Inference."
    ),
    version="3.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS (Vercel frontend için) ──────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,https://findik-dashboard.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def _reel_usd_to_tl(reel_usd: float, kur: float, yil: int = 2026) -> tuple[float, float]:
    cpi = US_CPI_TABLE[CPI_BAZ_YILI] / US_CPI_TABLE.get(yil, US_CPI_TABLE[CPI_BAZ_YILI])
    nom = reel_usd / cpi
    return nom, nom * kur


def _predict_single(row_dict: dict, kur: float, yil: int = 2026) -> tuple[float, float, float]:
    """Weighted ensemble tahmini: XGBoost + LightGBM + Ridge."""
    X = pd.DataFrame([row_dict])

    p_xgb = float(np.expm1(_state["xgb_m"].predict(X)[0]))
    p_lgb = float(np.expm1(_state["lgb_m"].predict(X)[0]))

    try:
        X_ridge = _state["ridge_scaler"].transform(X[list(row_dict.keys())])
        p_ridge = float(np.expm1(_state["ridge_m"].predict(X_ridge)[0]))
    except Exception:
        p_ridge = p_xgb  # fallback

    w = _state["weights"]
    reel = (
        w.get("XGBoost", 0.72)  * p_xgb +
        w.get("LightGBM", 0.0)  * p_lgb +
        w.get("Ridge", 0.28)    * p_ridge
    )
    nom, tl = _reel_usd_to_tl(reel, kur, yil)
    return reel, nom, tl


def _load_conformal_q_hat() -> float:
    cb_path = MODELS_DIR / "conformal_bounds.json"
    if cb_path.exists():
        try:
            with open(cb_path, "r") as f:
                return float(json.load(f).get("q_hat_relative", 0.10))
        except Exception:
            pass
    return 0.10


# ─── Pydantic Modeller ────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    usd_try: float = Field(44.0, ge=20.0, le=100.0, description="Nisan 2026 USD/TRY kuru")
    aylik_kur_artis: float = Field(0.008, ge=-0.05, le=0.10, description="Aylık kur artış oranı (0.008 = %0.8)")

class WhatIfRequest(BaseModel):
    usd_try: float = Field(44.0, ge=20.0, le=100.0)
    brent_petrol: Optional[float] = Field(None, ge=30.0, le=200.0)
    altin_ons: Optional[float] = Field(None, ge=500.0, le=5000.0)
    asgari_ucret: Optional[float] = Field(None, ge=5000.0, le=50000.0)
    rekolte_degisim_pct: float = Field(0.0, ge=-60.0, le=60.0, description="Rekolte değişim yüzdesi")
    ihracat_degisim_pct: float = Field(0.0, ge=-60.0, le=60.0)

class MonthlyPrediction(BaseModel):
    ay: int
    ay_adi: str
    kur: float
    reel_usd: float
    nominal_usd: float
    tl: float
    ci_low: float
    ci_high: float

class PredictResponse(BaseModel):
    predictions: list[MonthlyPrediction]
    q_hat: float
    model_info: dict

class WhatIfResponse(BaseModel):
    whatif_tl: float
    baz_tl: float
    delta_tl: float
    delta_pct: float
    whatif_reel_usd: float
    whatif_nominal_usd: float


# ─── Endpoint'ler ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistem"])
def health():
    """Sistem sağlık kontrolü."""
    return {
        "status": "ok",
        "model": "weighted_ensemble_v3.1",
        "data_rows": len(_state.get("df", [])),
    }


@app.get("/api/info", tags=["Model"])
def model_info():
    """Model ve veri bilgisi."""
    df: pd.DataFrame = _state.get("df", pd.DataFrame())
    return {
        "model":       "Weighted Ensemble (XGBoost + Ridge)",
        "version":     "3.1.0",
        "target":      TARGET,
        "data_rows":   int(len(df)),
        "data_start":  str(df["Tarih"].min().date()) if not df.empty else None,
        "data_end":    str(df["Tarih"].max().date()) if not df.empty else None,
        "features_n":  len(_state.get("sel_cols", [])),
        "weights":     _state.get("weights", {}),
        "config":      {
            "top_n_features": TOP_N_FEATURES,
            "cpi_base_year":  CPI_BAZ_YILI,
        },
    }


@app.post("/api/predict", response_model=PredictResponse, tags=["Tahmin"])
def predict(req: PredictRequest):
    """2026 Nisan–Aralık aylık fiyat tahminleri."""
    try:
        df        = _state["df"]
        sel_cols  = _state["sel_cols"]
        X_all     = _state["X_all"]
        q_hat     = _load_conformal_q_hat()

        drop_existing = [c for c in DROP_COLS if c in df.columns]
        X_df = df.drop(columns=drop_existing).select_dtypes(include=[np.number])

        last_row        = X_df[sel_cols].iloc[-1].to_dict()
        prev_usd        = float(df["Fiyat_USD_kg"].iloc[-1])
        prev2_usd       = float(df["Fiyat_USD_kg"].iloc[-2])
        prev3_usd       = float(df["Fiyat_USD_kg"].iloc[-3])
        prev_real_usd   = float(df["Fiyat_RealUSD_kg"].iloc[-1])
        prev3_real_usd  = float(df["Fiyat_RealUSD_kg"].iloc[-3])

        results = []
        for ay in range(4, 13):
            current_kur = req.usd_try * (1 + req.aylik_kur_artis) ** (ay - 4)
            row = dict(last_row)
            if "USD_Lag1"     in row: row["USD_Lag1"]     = prev_usd
            if "USD_Lag2"     in row: row["USD_Lag2"]     = prev2_usd
            if "USD_Lag3"     in row: row["USD_Lag3"]     = prev3_usd
            if "RealUSD_Lag1" in row: row["RealUSD_Lag1"] = prev_real_usd
            if "RealUSD_Lag3" in row: row["RealUSD_Lag3"] = prev3_real_usd

            reel, nom, tl = _predict_single(row, current_kur, 2026)
            ci_low  = round(tl * (1 - q_hat), 2)
            ci_high = round(tl * (1 + q_hat), 2)

            results.append(MonthlyPrediction(
                ay          = ay,
                ay_adi      = AYLAR_TR[ay],
                kur         = round(current_kur, 3),
                reel_usd    = round(reel, 3),
                nominal_usd = round(nom, 3),
                tl          = round(tl, 2),
                ci_low      = ci_low,
                ci_high     = ci_high,
            ))

            prev3_usd      = prev2_usd
            prev2_usd      = prev_usd
            prev_usd       = nom
            prev3_real_usd = prev_real_usd
            prev_real_usd  = reel

        return PredictResponse(
            predictions = results,
            q_hat       = q_hat,
            model_info  = {
                "algorithm": "Weighted Ensemble",
                "weights": _state["weights"],
                "test_mape": 9.05,
                "test_r2":   0.453,
            },
        )

    except Exception as exc:
        logger.error(f"/api/predict hatası: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/whatif", response_model=WhatIfResponse, tags=["Tahmin"])
def whatif(req: WhatIfRequest):
    """What-If senaryo analizi."""
    try:
        sel_cols = _state["sel_cols"]
        X_all    = _state["X_all"]

        base_row = X_all[sel_cols].iloc[-1].to_dict()
        wi_row   = dict(base_row)

        if req.brent_petrol is not None and "Brent_Petrol_Kapanis" in wi_row:
            wi_row["Brent_Petrol_Kapanis"] = req.brent_petrol
        if req.altin_ons is not None and "Altin_Ons_Kapanis" in wi_row:
            wi_row["Altin_Ons_Kapanis"] = req.altin_ons
        if req.asgari_ucret is not None and "Asgari_Ucret_TL" in wi_row:
            wi_row["Asgari_Ucret_TL"] = req.asgari_ucret
        if "USD_TRY_Kapanis" in wi_row:
            wi_row["USD_TRY_Kapanis"] = req.usd_try

        for col in ["Uretim_Ton", "Dunya_Uretim_Ton"]:
            if col in wi_row and wi_row[col] and wi_row[col] != 0:
                wi_row[col] *= (1 + req.rekolte_degisim_pct / 100)
        for col in ["Ihracat_Ton", "Ihracat_Miktar_Ton"]:
            if col in wi_row and wi_row[col] and wi_row[col] != 0:
                wi_row[col] *= (1 + req.ihracat_degisim_pct / 100)

        df_last_kur = float(_state["df"]["Fiyat_USD_kg"].iloc[-1])
        wi_reel, wi_nom, wi_tl     = _predict_single(wi_row, req.usd_try, 2026)
        base_reel, base_nom, base_tl = _predict_single(base_row, 44.0, 2026)

        delta_tl  = wi_tl - base_tl
        delta_pct = (delta_tl / base_tl * 100) if base_tl else 0

        return WhatIfResponse(
            whatif_tl         = round(wi_tl, 2),
            baz_tl            = round(base_tl, 2),
            delta_tl          = round(delta_tl, 2),
            delta_pct         = round(delta_pct, 2),
            whatif_reel_usd   = round(wi_reel, 4),
            whatif_nominal_usd= round(wi_nom, 4),
        )

    except Exception as exc:
        logger.error(f"/api/whatif hatası: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/scores", tags=["Model"])
def model_scores():
    """Tüm model test seti performans skorları."""
    scores_path = MODELS_DIR / "all_model_scores.json"
    if scores_path.exists():
        with open(scores_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model skorları bulunamadı.")


@app.get("/api/tmo", tags=["TMO"])
def tmo_prediction():
    """TMO 2026 taban fiyat tahmini ve CI."""
    tmo_path = MODELS_DIR / "tmo_prediction_2026.json"
    if tmo_path.exists():
        with open(tmo_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="TMO tahmini bulunamadı.")


@app.get("/api/causal", tags=["Analiz"])
def causal_effect():
    """Double ML nedensel etki: kur → fiyat."""
    causal_path = MODELS_DIR / "causal_effect.json"
    if causal_path.exists():
        with open(causal_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Nedensel etki verisi bulunamadı.")


@app.get("/api/weather", tags=["Veri"])
def weather_data():
    """Karadeniz hava durumu & iklim riski (3 aylık öngörü)."""
    hava_path = BASE_DIR / "data" / "processed" / "hava_durumu_3aylik.json"
    if hava_path.exists():
        with open(hava_path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Hava durumu verisi bulunamadı.")


@app.get("/api/history", tags=["Veri"])
def historical_prices(months: int = 36):
    """Son N aylık gerçek serbest piyasa fiyatları."""
    df: pd.DataFrame = _state.get("df", pd.DataFrame())
    if df.empty:
        raise HTTPException(status_code=503, detail="Veri henüz yüklenmedi.")

    tail = df.tail(months).copy()
    cols = ["Tarih", "Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "Fiyat_RealUSD_kg"]
    available = [c for c in cols if c in tail.columns]
    tail = tail[available].copy()
    tail["Tarih"] = tail["Tarih"].dt.strftime("%Y-%m-%d")
    return tail.fillna(0).to_dict(orient="records")

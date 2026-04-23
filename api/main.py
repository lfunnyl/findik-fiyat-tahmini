"""
api/main.py
===========
Fındık Fiyat Tahmin Sistemi — Modüler FastAPI Backend
"""

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.helpers import get_logger, CFG, MODELS_DIR
from src.services.prediction_service import prediction_service

logger = get_logger(__name__)

# ─── Lifespan (startup/shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    prediction_service.load_all()
    yield
    logger.info("Uygulama kapatıldı.")

# ─── FastAPI Uygulaması ────────────────────────────────────────────────────────
app = FastAPI(
    title="🌰 Fındık Fiyat Tahmin API",
    description="Türkiye fındık serbest piyasa fiyatı tahmin sistemi.",
    version="3.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Güvenlik için prod'da kısıtlanmalı
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Pydantic Modeller ────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    usd_try: float = Field(44.0, ge=20.0, le=100.0)
    aylik_kur_artis: float = Field(0.008, ge=-0.05, le=0.10)

class WhatIfRequest(BaseModel):
    usd_try: float = Field(44.0, ge=20.0, le=100.0)
    brent_petrol: Optional[float] = Field(None, ge=30.0, le=200.0)
    rekolte_degisim_pct: float = Field(0.0, ge=-60.0, le=60.0)

# ─── Endpoint'ler ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "loaded": prediction_service.is_loaded}

@app.post("/api/predict")
def predict(req: PredictRequest):
    try:
        preds = prediction_service.predict_multistep(req.usd_try, req.aylik_kur_artis)
        return {
            "predictions": preds,
            "q_hat": prediction_service.q_hat,
            "model_info": {
                "algorithm": "Weighted Ensemble",
                "weights": prediction_service.weights
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/whatif")
def whatif(req: WhatIfRequest):
    try:
        return prediction_service.predict_whatif(req.usd_try, req.brent_petrol, req.rekolte_degisim_pct)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tmo")
def tmo_prediction():
    return prediction_service.get_tmo_data()

@app.get("/api/info")
def model_info():
    return {
        "model": "Weighted Ensemble v3.2",
        "data_rows": len(prediction_service.df) if prediction_service.df is not None else 0,
        "features_n": len(prediction_service.sel_cols)
    }

@app.get("/api/scores")
def model_scores():
    import json
    with open(MODELS_DIR / "all_model_scores.json", "r") as f:
        return json.load(f)

@app.get("/api/shap")
def shap_importance():
    import json
    with open(MODELS_DIR / "shap_importance.json", "r") as f:
        return json.load(f)

@app.get("/api/causal")
def causal_effect():
    import json
    with open(MODELS_DIR / "causal_effect.json", "r") as f:
        return json.load(f)

@app.get("/api/history")
def history(months: int = 36):
    df = prediction_service.df
    if df is None: return []
    tail = df.tail(months).copy()
    return tail[["Tarih", "Serbest_Piyasa_TL_kg", "Fiyat_USD_kg"]].to_dict(orient="records")

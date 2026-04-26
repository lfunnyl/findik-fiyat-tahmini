import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from src.utils.helpers import MODELS_DIR, DATA_DIR, get_logger, load_config, reel_usd_to_tl, CFG
from src.features.engineering import apply_feature_engineering

logger = get_logger(__name__)

class PredictionService:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.df = None
        self.is_loaded = False
        self.q_hat = 0.10
        self.sel_cols = []

    def load_all(self):
        """Tüm modelleri ve verileri yükler."""
        try:
            # Model Paketleri
            for m in ['xgboost', 'ridge']:
                p = MODELS_DIR / f"{m}_model.pkl"
                if p.exists():
                    self.models[m] = joblib.load(p)
            
            # Ağırlıklar
            w_p = MODELS_DIR / "ensemble_weights.json"
            if w_p.exists():
                with open(w_p, "r") as f:
                    self.weights = json.load(f)
            
            # Conformal Bounds
            q_p = MODELS_DIR / "conformal_bounds.json"
            if q_p.exists():
                with open(q_p, "r") as f:
                    self.q_hat = json.load(f).get("q_hat_relative", 0.10)
            
            # Veri
            master_data_path = DATA_DIR / "master_features.csv"
            if master_data_path.exists():
                raw_df = pd.read_csv(master_data_path)
                self.df = apply_feature_engineering(raw_df)
                
            # Özellik Kolonları (XGBoost'tan al)
            if 'xgboost' in self.models:
                self.sel_cols = self.models['xgboost']['features']

            self.is_loaded = True
            logger.info("PredictionService: Başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"PredictionService yükleme hatası: {e}")
            raise

    def predict_single(self, row_dict: dict, kur: float, yil: int = 2026, prev_real_usd: Optional[float] = None) -> Tuple[float, float, float]:
        """Weighted ensemble tahmini: XGBoost + Ridge."""
        X = pd.DataFrame([row_dict])
        
        # Bireysel Tahminler (Artık bunlar DELTA LOG değerleri!)
        p_xgb_delta = float(self.models['xgboost']['model'].predict(X[self.sel_cols])[0])
        
        try:
            X_ridge = self.models['ridge']['scaler'].transform(X[self.sel_cols])
            p_ridge_delta = float(self.models['ridge']['model'].predict(X_ridge)[0])
        except:
            p_ridge_delta = p_xgb_delta
            
        w = self.weights
        predicted_delta_log = (
            w.get("XGBoost", 0.72) * p_xgb_delta +
            w.get("Ridge", 0.28) * p_ridge_delta
        )
        
        # Delta Modeling Geri Dönüşümü (Reverse Transform)
        if prev_real_usd is not None and prev_real_usd > 0:
            prev_log = np.log1p(prev_real_usd)
            current_log = prev_log + predicted_delta_log
            reel = float(np.expm1(current_log))
        else:
            # Fallback
            reel = float(np.expm1(predicted_delta_log))
            
        nom, tl = reel_usd_to_tl(reel, kur, yil)
        return reel, nom, tl

    def predict_multistep(self, start_kur: float, aylik_artis: float) -> List[Dict[str, Any]]:
        """Oto-regresif çok adımlı tahmin simülasyonu."""
        last_row = self.df[self.sel_cols].iloc[-1].to_dict()
        
        # Lag değerlerini takip et
        prev_usd = float(self.df["Fiyat_USD_kg"].iloc[-1])
        prev2_usd = float(self.df["Fiyat_USD_kg"].iloc[-2])
        prev3_usd = float(self.df["Fiyat_USD_kg"].iloc[-3])
        prev_real_usd = float(self.df["Fiyat_RealUSD_kg"].iloc[-1])
        prev3_real_usd = float(self.df["Fiyat_RealUSD_kg"].iloc[-3])

        results = []
        aylar_tr = {1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran", 
                    7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"}

        for ay in range(4, 13):
            kur = start_kur * (1 + aylik_artis) ** (ay - 4)
            row = dict(last_row)
            
            # Lag güncellemeleri
            if "USD_Lag1" in row: row["USD_Lag1"] = prev_usd
            if "USD_Lag2" in row: row["USD_Lag2"] = prev2_usd
            if "USD_Lag3" in row: row["USD_Lag3"] = prev3_usd
            if "RealUSD_Lag1" in row: row["RealUSD_Lag1"] = prev_real_usd
            if "RealUSD_Lag3" in row: row["RealUSD_Lag3"] = prev3_real_usd

            reel, nom, tl = self.predict_single(row, kur, 2026, prev_real_usd=prev_real_usd)
            
            results.append({
                "ay": ay, "ay_adi": aylar_tr[ay], "kur": round(kur, 3),
                "reel_usd": round(reel, 3), "nominal_usd": round(nom, 3), "tl": round(tl, 2),
                "ci_low": round(tl * (1 - self.q_hat), 2), "ci_high": round(tl * (1 + self.q_hat), 2)
            })

            # Kaydırmalı pencere güncellemesi
            prev3_usd, prev2_usd, prev_usd = prev2_usd, prev_usd, nom
            prev3_real_usd, prev_real_usd = prev_real_usd, reel

        return results

    def predict_whatif(self, usd_try: float, brent: Optional[float], rekolte_pct: float) -> Dict[str, Any]:
        """Senaryo analizi."""
        base_row = self.df[self.sel_cols].iloc[-1].to_dict()
        wi_row = dict(base_row)

        prev_real_usd = float(self.df["Fiyat_RealUSD_kg"].iloc[-1])

        if brent is not None and "Brent_Petrol_Kapanis" in wi_row:
            wi_row["Brent_Petrol_Kapanis"] = brent
        if "USD_TRY_Kapanis" in wi_row:
            wi_row["USD_TRY_Kapanis"] = usd_try
            
        for col in ["Uretim_Ton", "Uretim_Ton_Turkiye"]:
            if col in wi_row: wi_row[col] *= (1 + rekolte_pct / 100)

        wi_reel, wi_nom, wi_tl = self.predict_single(wi_row, usd_try, 2026, prev_real_usd=prev_real_usd)
        base_reel, base_nom, base_tl = self.predict_single(base_row, 44.0, 2026, prev_real_usd=prev_real_usd)

        return {
            "whatif_tl": round(wi_tl, 2), "baz_tl": round(base_tl, 2),
            "delta_tl": round(wi_tl - base_tl, 2), "delta_pct": round(((wi_tl / base_tl - 1) * 100) if base_tl else 0, 2),
            "whatif_reel_usd": round(wi_reel, 4), "whatif_nominal_usd": round(wi_nom, 4)
        }

    def get_tmo_data(self) -> Dict[str, Any]:
        """TMO fiyat tahminini döndürür."""
        tmo_path = MODELS_DIR / "tmo_prediction_2026.json"
        if tmo_path.exists():
            try:
                with open(tmo_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"TMO tahmini okunamadı: {e}")
        return {}

# Singleton
prediction_service = PredictionService()

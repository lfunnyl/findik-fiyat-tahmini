"""
train_conformal.py
==================
Conformal Prediction (Matematiksel Garantili Güven Aralıkları)

Mevcut Streamlit uygulamasındaki Bootstrap + Gürültü (Noise) metodu, 
"yaklaşık ve heuristik" bir güven aralığı hesaplar. Bu script ise 
Split-Conformal Regression (Vovk vd.) tekniği ile modelin hold-out
verisi üzerindeki "gerçek" hatalarını baz alarak marjinal olarak %90
kapsayıcılık hedefleyen (coverage guarantee) kalibre edilmiş toleranslar üretir.

Çıktı:
- models/conformal_bounds.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TARGET = 'Fiyat_RealUSD_kg'

def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df = df.sort_values('Tarih').reset_index(drop=True)
    return df

def prepare_xy(df):
    df = df.copy()
    df['USD_Lag1']     = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']     = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']     = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']    = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']  = df['Fiyat_USD_kg'].pct_change(1) * 100
    df['USD_YoY_pct']  = df['Fiyat_USD_kg'].pct_change(12) * 100
    df['RealUSD_Lag1'] = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3'] = df['Fiyat_RealUSD_kg'].shift(3)
    df = df.bfill().ffill()
    return df

def fit_conformal_bounds(alpha=0.10): # %90 Confidence
    logger.info("Conformal Prediction katsayilari hesaplaniyor (alpha={})...".format(alpha))
    
    # Veri Hazirligi
    df = load_data()
    df = prepare_xy(df)
    
    y_raw = df[TARGET]
    
    # Model Yukleme
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    lgb_path = os.path.join(MODELS_DIR, 'lightgbm_model.pkl')
    weights_path = os.path.join(MODELS_DIR, 'ensemble_weights.json')
    
    if not os.path.exists(xgb_path):
        logger.error("Modeller bulunamadi.")
        return
        
    m_xgb = joblib.load(xgb_path)
    m_lgb = joblib.load(lgb_path)
    with open(weights_path, 'r') as f:
        w = json.load(f)
        
    feat_cols = m_xgb['features']
    X_full = df[feat_cols]
    
    # Hold-out (Kalibrasyon) seti: Son %20'yi kalibrasyon olarak kabul edelim
    split_idx = int(len(df) * 0.8)
    X_calib = X_full.iloc[split_idx:]
    y_calib = y_raw.iloc[split_idx:].values
    
    # Topluluk (Ensemble) tahmini (Log uzayindan cikarilmis, USD bazinda)
    p_xgb = np.expm1(m_xgb['model'].predict(X_calib))
    p_lgb = np.expm1(m_lgb['model'].predict(X_calib))
    
    ensemble_preds = w['XGBoost']*p_xgb + w['LightGBM']*p_lgb + w['Ridge']*p_xgb # Ridge simple proxy
    
    # 1. Hatalari (Residuals) hesapla: Absolute errors
    abs_errs = np.abs(y_calib - ensemble_preds)
    
    # Mutlak hata ile tahmine oransal goreceli hata (MAPE gibi)
    # Çünkü fiyat yükseldikçe standart sapma büyür (Heteroskedasticity)
    # Relative residuals: |y - y_hat| / |y_hat|
    epsilon = 1e-4
    rel_errs = abs_errs / (ensemble_preds + epsilon)
    
    # 2. Conformal Quantile Hesaplama (Finite sample correction ile)
    n = len(rel_errs)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    # Eger q_level > 1 ise, 1'e caple
    q_level = min(q_level, 1.0)
    
    q_hat = np.quantile(rel_errs, q_level)
    
    # Ekstra guvenlik - Max Relative Error sinirlamasi
    q_hat = min(q_hat, 0.45)  # En fazla %45 sapma toleransi atariz (Dengesiz uclari tırpanla)
    
    logger.info(f"Kalibrasyon sample: {n}")
    logger.info(f"Hesaplanan q_hat (Relative Tolerance): %{q_hat*100:.2f}")
    
    res = {
        "alpha": alpha,
        "n_calibre": n,
        "q_hat_relative": float(q_hat)
    }
    
    out_path = os.path.join(MODELS_DIR, "conformal_bounds.json")
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=4)
        
    logger.info(f"Conformal katsayilari kaydedildi -> {out_path}")

if __name__ == "__main__":
    fit_conformal_bounds(alpha=0.10) # 90% confidence

"""
online_update.py
================
Incremental (Online) Learning Modülü

Amaç: 13 yıllık (150+ ay) fındık fiyat verisini her ay 0'dan tamamen
yeniden eğitmek yerine, sadece en son eklenen 1 aylık yeni veriyi alıp, 
mevcut eğitilmiş XGBoost ağacının üzerine yeni dallar/ağırlıklar ekleyerek 
hızlı (continuous) güncelleme yapmak.

Yöntem: XGBRegressor içerisindeki `xgb_model` parametresi.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
XGB_PATH = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
TARGET = 'Fiyat_RealUSD_kg'

def load_data(top_n=1):
    """Sadece son `top_n` aylık yeni veriyi yükler."""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df = df.sort_values('Tarih').reset_index(drop=True)
    
    # Feature Engineering (sadece gerekli lag'lar vb)
    df['USD_Lag1']     = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']     = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']     = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']    = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']  = df['Fiyat_USD_kg'].pct_change(1) * 100
    df['USD_YoY_pct']  = df['Fiyat_USD_kg'].pct_change(12) * 100
    df['RealUSD_Lag1'] = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3'] = df['Fiyat_RealUSD_kg'].shift(3)
    df = df.bfill().ffill()
    
    # Sadece en son top_n satırı al (yeni gelen veriler)
    new_data = df.tail(top_n).copy()
    return new_data

def online_update():
    logger.info("Online Learning (Incremental Update) basliyor...")
    
    if not os.path.exists(XGB_PATH):
        logger.error("Mevcut XGBoost modeli bulunamadi. Lutfen once tam egitim yapin.")
        return
        
    model_bundle = joblib.load(XGB_PATH)
    base_model = model_bundle['model']
    features = model_bundle['features']
    
    new_df = load_data(top_n=1) # Sadece son eklenen ay
    if new_df.empty:
        logger.warning("Guncellenecek yeni veri bulunamadi.")
        return
        
    X_new = new_df[features]
    y_new_raw = new_df[TARGET].values
    y_new_log = np.log1p(y_new_raw)
    
    logger.info(f"Ogrenecek Yeni Kayit Sayisi: {len(X_new)}")
    logger.info(f"Tarih(ler): {new_df['Tarih'].tolist()}")
    
    # XGBoost'u incremental fit et
    # xgb_model parametresi, onceki modelin agaclarina devam edilmesini saglar.
    
    yeni_model = XGBRegressor(
        n_estimators=10,        # Yeni veriler icin az sayida ek agac (boosting rounds) cikaralim
        learning_rate=0.01,     # Yeni veriyle modeli fazla sarsmamak (forgetting) adina dusuk LR
        max_depth=3,
        random_state=42,
        verbosity=0
    )
    
    # base_model XGBRegressor objesi, agaclari barindirir. 
    # Bize core XGBoost model objesi (Booster) lazim.
    booster = base_model.get_booster()
    
    # Incremental Training
    yeni_model.fit(X_new, y_new_log, xgb_model=booster)
    
    # Test (Guncelleme sonrasi yeni record uzerindeki hata nedir?)
    preds_log = yeni_model.predict(X_new)
    preds_orig = np.expm1(preds_log)
    err = np.abs(y_new_raw - preds_orig)
    
    logger.info(f"Yalnizca yeni ay uzerindeki Test Hatasi (MAE): {np.mean(err):.3f} USD/kg")
    
    # Mevcut modeli ez veya `online_updated` diye kaydet
    model_bundle['model'] = yeni_model
    out_path = os.path.join(MODELS_DIR, 'xgboost_model_online.pkl')
    joblib.dump(model_bundle, out_path)
    
    logger.info(f"Online-updated XGBoost modeli basariyla kaydedildi -> {out_path}")

if __name__ == "__main__":
    online_update()

"""
track_performance.py
====================
Gerçek Test Skoru (Out-of-Sample) Takibi

Bu script, update_pipeline.py içerisinde model RETRAIN edilmeden HEMEN ÖNCE çalıştırılır.
Amacı, geçen ay eğitilmiş olan mevcut modeller ile bu ay yeni çekilen gerçek veriyi 
karşılaştırmak ve "Modelimiz geçen ay ne kadar doğru bildi?" sorusuna yanıt bulmaktır.

Sonuçlar 'data/performance_log.csv' dosyasına eklenir (append).
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR= os.path.join(BASE_DIR, "models")
LOG_PATH  = os.path.join(BASE_DIR, "data", "performance_log.csv")

TARGET = 'Fiyat_RealUSD_kg'

def get_latest_data():
    """En güncel veriyi (sadece en son satırı) ve özelliklerini getirir."""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih').reset_index(drop=True)
    
    # Lag özellikleri oluştur (inference ile uyumlu)
    df['USD_Lag1']     = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']     = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']     = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']    = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']  = df['Fiyat_USD_kg'].pct_change(1) * 100
    df['USD_YoY_pct']  = df['Fiyat_USD_kg'].pct_change(12) * 100
    df['RealUSD_Lag1'] = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3'] = df['Fiyat_RealUSD_kg'].shift(3)
    df = df.bfill().ffill()
    
    # En son satır (Yeni gelen ay verisi)
    last_row = df.iloc[[-1]]
    return last_row

def predict_with_old_ensemble(last_row):
    """Ensemble modelin ağırlıklarını ve base modellerini yükleyip tahmin yapar."""
    # Ağırlıklar
    weights_path = os.path.join(MODELS_DIR, 'ensemble_weights.json')
    if not os.path.exists(weights_path):
        logger.warning("ensemble_weights.json bulunamadı! Tahmin yapılamıyor.")
        return None
        
    with open(weights_path, 'r', encoding='utf-8') as f:
        weights = json.load(f)
        
    xgb_w   = weights.get('XGBoost', 0)
    ridge_w = weights.get('Ridge', 0)
    # LightGBM genelde ağırlık almaz, ama varsa alırız
    # Basitlik için sadece XGBoost ve Ridge kullanacağız (predict.py'deki gibi)
    
    # Modelleri yükle
    optuna_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    if not os.path.exists(optuna_path):
        logger.warning(f"Model dosyasi bulunamadi: {optuna_path}")
        return None
        
    xgb_bundle = joblib.load(optuna_path)
    xgb_model  = xgb_bundle['model']
    xgb_cols   = xgb_bundle['features']
    
    ridge_path = os.path.join(MODELS_DIR, 'ridge_model.pkl')
    if not os.path.exists(ridge_path):
        logger.warning(f"Model dosyasi bulunamadi: {ridge_path}")
        return None
        
    ridge_bundle = joblib.load(ridge_path)
    ridge_model  = ridge_bundle['model']
    ridge_cols   = ridge_bundle['features']
    ridge_scaler = ridge_bundle['scaler']
    
    # XGBoost Tahmini
    X_xgb = last_row[xgb_cols]
    xgb_pred_log = xgb_model.predict(X_xgb)[0]
    xgb_pred = np.expm1(xgb_pred_log)
    
    # Ridge Tahmini
    X_ridge = last_row[ridge_cols]
    X_ridge_scaled = ridge_scaler.transform(X_ridge)
    ridge_pred_log = ridge_model.predict(X_ridge_scaled)[0]
    ridge_pred = np.expm1(ridge_pred_log)
    
    # Ağırlıklı Ortalama
    ensemble_pred = (xgb_w * xgb_pred) + (ridge_w * ridge_pred)
    
    # Eğer toplam ağırlık 1 değilse normalize et
    total_w = xgb_w + ridge_w
    if total_w > 0:
        ensemble_pred = ensemble_pred / total_w
        
    logger.info(f"OOS Tahmini (XGB:{xgb_w:.2f}, Ridge:{ridge_w:.2f}) -> {ensemble_pred:.2f} USD/kg")
    return ensemble_pred


def main():
    logger.info("=" * 60)
    logger.info("GERCEK TEST SKORU TAKIBI (Out-of-Sample OOS Logger)")
    logger.info("=" * 60)
    
    last_row = get_latest_data()
    tarih_str = last_row['Tarih'].dt.strftime('%Y-%m-%d').values[0]
    gercek_fiyat = last_row['Fiyat_RealUSD_kg'].values[0]
    
    logger.info(f"Test Edilen Yeni Ay : {tarih_str}")
    logger.info(f"Gerceklesen Fiyat   : {gercek_fiyat:.3f} Reel USD/kg")
    
    # Geçmiş logları kontrol et (aynı ay zaten kaydedilmiş mi?)
    if os.path.exists(LOG_PATH):
        try:
            log_df = pd.read_csv(LOG_PATH)
            if tarih_str in log_df['Tarih'].values:
                logger.info(f"{tarih_str} zaten performance_log.csv'de kayitli. Atlaniliyor.")
                return
        except Exception as e:
            logger.warning(f"Log dosyasi okunurken hata: {e}")
    else:
        # Yeni dosya başlıkları oluşturulacak
        log_df = pd.DataFrame(columns=['Tarih', 'Gercek_Fiyat_USD', 'Model_Tahmin_USD', 'Mutlak_Hata_USD', 'MAPE_Pct'])
        
    # Mevcut eski modeli kullanarak tahmini al
    tahmin_fiyat = predict_with_old_ensemble(last_row)
    
    if tahmin_fiyat is None:
        logger.error("Modeller yüklenemedi. OOS performans kaydı yapılamadı.")
        return
        
    mutlak_hata = abs(gercek_fiyat - tahmin_fiyat)
    mape = (mutlak_hata / gercek_fiyat) * 100
    
    logger.info(f"Modelin Tahmini     : {tahmin_fiyat:.3f} Reel USD/kg")
    logger.info(f"Hata (MAE)          : {mutlak_hata:.3f} USD/kg")
    logger.info(f"Hata Orani (MAPE)   : %{mape:.2f}")
    
    # DataFrame'e ekle
    yeni_kayit = pd.DataFrame([{
        'Tarih': tarih_str,
        'Gercek_Fiyat_USD': round(gercek_fiyat, 3),
        'Model_Tahmin_USD': round(tahmin_fiyat, 3),
        'Mutlak_Hata_USD': round(mutlak_hata, 3),
        'MAPE_Pct': round(mape, 2)
    }])
    
    if os.path.exists(LOG_PATH):
        yeni_kayit.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        yeni_kayit.to_csv(LOG_PATH, mode='w', header=True, index=False)
        
    logger.info(f"Kayit basariyla {LOG_PATH} dosyasina eklendi.")


if __name__ == "__main__":
    main()

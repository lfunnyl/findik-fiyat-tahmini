"""
train_multistep.py
==================
Direct Multi-Step Forecasting (1, 3, 6 Aylık Tahmin Modelleri)

Mevcut recursive (özyinelemeli) tahmin modeli uzun vadede hata birikimine yol açabileceğinden,
bu script ile doğrudan t+3 ve t+6 aylarını hedefleyen modeller eğitilir.

Strateji:
  - 1-Ay (1M): target = y.shift(0)
  - 3-Ay (3M): target = y.shift(-2)
  - 6-Ay (6M): target = y.shift(-5)
  Her bir horizon (ufuk) için ayrı bir XGBoost modeli eğitilerek kaydedilir.
"""

import os
import warnings
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

TARGET = 'Fiyat_RealUSD_kg'
TOP_N_FEATURES = 20

DROP_COLS = [
    TARGET, 'Serbest_Piyasa_TL_kg', 'Fiyat_USD_kg', 'US_CPI_Carpani',
    'Tarih', 'Yil_Ay', 'Hasat_Donemi',
    'TMO_Giresun_TL_kg', 'TMO_Levant_TL_kg',
    'Fiyat_Lag1', 'Fiyat_Lag2', 'Fiyat_Lag3', 'Fiyat_Lag12',
    'Fiyat_Degisim_1A_Pct', 'Fiyat_Degisim_3A_Pct',
    'Fiyat_bolu_AsgariUcret_Orani', 'Yil', 'Sezon_Yili',
]

def load_data():
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih').reset_index(drop=True)
    return df

def prepare_base_features(df):
    df = df.copy()
    # Bazı feature'lar (train_model.py ile aynı)
    df['USD_Lag1']      = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']      = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']      = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']     = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']   = df['Fiyat_USD_kg'].pct_change(1) * 100
    df['USD_YoY_pct']   = df['Fiyat_USD_kg'].pct_change(12) * 100
    df['RealUSD_Lag1']  = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3']  = df['Fiyat_RealUSD_kg'].shift(3)
    df = df.bfill().ffill()

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_raw = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    target_raw = df[TARGET]
    
    return X_raw, target_raw

def select_features(X_train, y_train, top_n=TOP_N_FEATURES):
    corr = X_train.corrwith(y_train).abs().dropna()
    selected = corr.nlargest(top_n).index.tolist()
    return selected

def train_direct_model(X, y_raw, horizon_shift, model_name):
    """
    horizon_shift: e.g., 0 for 1M (t+1), -2 for 3M (t+3), -5 for 6M (t+6)
    """
    logger.info(f"\n{'='*50}\nEğitiliyor: {model_name} (Shift: {horizon_shift})\n{'='*50}")
    
    # Target shift (Log uzayı)
    y_shifted = y_raw.shift(horizon_shift)
    
    # NaN olan kısımları düşür (Son satırlar)
    valid_idx = y_shifted.dropna().index
    X_valid = X.loc[valid_idx]
    y_valid_log = np.log1p(y_shifted.loc[valid_idx])
    y_valid_raw = y_shifted.loc[valid_idx]
    
    split_idx = int(len(X_valid) * 0.8)
    X_train = X_valid.iloc[:split_idx]
    X_test  = X_valid.iloc[split_idx:]
    y_train_log = y_valid_log.iloc[:split_idx]
    y_test_raw  = y_valid_raw.iloc[split_idx:]
    
    # Feature Selection
    sel_cols = select_features(X_train, y_train_log, TOP_N_FEATURES)
    X_train_sel = X_train[sel_cols]
    X_test_sel  = X_test[sel_cols]
    
    # Model (XGBoost)
    model = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbosity=0
    )
    
    model.fit(X_train_sel, y_train_log)
    
    # Test Evaluation
    preds_log = model.predict(X_test_sel)
    preds_orig = np.expm1(preds_log)
    
    try:
        mae = mean_absolute_error(y_test_raw, preds_orig)
        r2 = r2_score(y_test_raw, preds_orig)
        mape = np.mean(np.abs((y_test_raw.values - preds_orig) / np.where(y_test_raw.values == 0, 1, y_test_raw.values))) * 100
        logger.info(f"{model_name} Test Sonucu -> MAE: {mae:.3f} RealUSD | R²: {r2:.3f} | MAPE: %{mape:.2f}")
    except Exception as e:
        logger.warning(f"Metrik hesaplanamadı: {e}")
        
    # Tüm veriyle yeniden fit edip kaydetme (Production Deploy)
    X_full_sel = X_valid[sel_cols]
    model_full = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.7,
        random_state=42, verbosity=0
    )
    model_full.fit(X_full_sel, y_valid_log)
    
    save_dict = {'model': model_full, 'features': sel_cols, 'horizon': model_name}
    p = os.path.join(MODELS_DIR, f"{model_name.lower()}.pkl")
    joblib.dump(save_dict, p)
    logger.info(f"Model kaydedildi: {p}")

def main():
    df = load_data()
    X_raw, target_raw = prepare_base_features(df)
    
    # 1. Ay (t+1) Model: shift = 0
    train_direct_model(X_raw, target_raw, horizon_shift=0, model_name="multistep_1m")
    
    # 3. Ay (t+3) Model: shift = -2
    train_direct_model(X_raw, target_raw, horizon_shift=-2, model_name="multistep_3m")
    
    # 6. Ay (t+6) Model: shift = -5
    train_direct_model(X_raw, target_raw, horizon_shift=-5, model_name="multistep_6m")
    
if __name__ == "__main__":
    main()

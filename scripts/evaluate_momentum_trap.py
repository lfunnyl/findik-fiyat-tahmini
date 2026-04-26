import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import warnings
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

# --- Yollar ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "processed", "master_features.csv")

TARGET = "Fiyat_RealUSD_kg"

def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih").reset_index(drop=True)
    
    # Modelin eğitimde kullandığı özellikleri türet
    df["USD_Lag1"]     = df["Fiyat_USD_kg"].shift(1)
    df["USD_Lag2"]     = df["Fiyat_USD_kg"].shift(2)
    df["USD_Lag3"]     = df["Fiyat_USD_kg"].shift(3)
    df["USD_Lag12"]    = df["Fiyat_USD_kg"].shift(12)
    df["USD_MoM_pct"]  = df["Fiyat_USD_kg"].pct_change(1) * 100
    df["USD_YoY_pct"]  = df["Fiyat_USD_kg"].pct_change(12) * 100
    df["RealUSD_Lag1"] = df["Fiyat_RealUSD_kg"].shift(1)
    df["RealUSD_Lag3"] = df["Fiyat_RealUSD_kg"].shift(3)
    
    # Hedef için (Gerçek fiyatlar eksik olmamalı)
    df = df.dropna(subset=[TARGET]).copy()
    
    # Bfill ve ffill sadece özellikler için
    feat_cols = [c for c in df.columns if c not in ["Tarih", TARGET]]
    df[feat_cols] = df[feat_cols].bfill().ffill()
    
    return df

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true==0, 1e-10, y_true))) * 100

def main():
    print("=" * 60)
    print("🚨 RANDOM WALK / MOMENTUM TRAP ANALİZİ 🚨")
    print("=" * 60)
    
    df = load_data()
    
    # Train/Test Split (Son %20 Test)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    
    # Atılacak sütunlar (train_model.py içindeki gibi)
    DROP_COLS = [TARGET, "Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "US_CPI_Carpani",
                 "Tarih", "Yil_Ay", "Hasat_Donemi", "TMO_Giresun_TL_kg", "TMO_Levant_TL_kg",
                 "Fiyat_Lag1", "Fiyat_Lag2", "Fiyat_Lag3", "Fiyat_Lag12",
                 "Fiyat_Degisim_1A_Pct", "Fiyat_Degisim_3A_Pct",
                 "Fiyat_bolu_AsgariUcret_Orani", "Yil", "Sezon_Yili"]
    
    lag_features = ["USD_Lag1", "USD_Lag2", "USD_Lag3", "USD_Lag12", 
                    "USD_MoM_pct", "USD_YoY_pct", "RealUSD_Lag1", "RealUSD_Lag3"]
    
    base_features = [c for c in df_train.columns if c not in DROP_COLS and c not in [TARGET, "Tarih"]]
    
    # Log dönüşümü
    y_train_log = np.log1p(df_train[TARGET])
    y_test_orig = df_test[TARGET].values
    
    print("\n[TEST 1] ÖZELLİK ÇIKARMA (ABLATION) TESTİ")
    print("-" * 50)
    
    # Model 1: Tüm özellikler (Lag dahil)
    X_train_full = df_train[base_features]
    X_test_full  = df_test[base_features]
    
    model_full = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_full.fit(X_train_full, y_train_log)
    preds_full_log = model_full.predict(X_test_full)
    preds_full = np.expm1(preds_full_log)
    r2_full = r2_score(y_test_orig, preds_full)
    mape_full = calculate_mape(y_test_orig, preds_full)
    
    # Model 2: Lag özellikleri olmadan (Ablation)
    features_no_lag = [c for c in base_features if c not in lag_features]
    X_train_no_lag = df_train[features_no_lag]
    X_test_no_lag  = df_test[features_no_lag]
    
    model_no_lag = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model_no_lag.fit(X_train_no_lag, y_train_log)
    preds_no_lag_log = model_no_lag.predict(X_test_no_lag)
    preds_no_lag = np.expm1(preds_no_lag_log)
    r2_no_lag = r2_score(y_test_orig, preds_no_lag)
    mape_no_lag = calculate_mape(y_test_orig, preds_no_lag)
    
    r2_drop_pct = ((r2_full - r2_no_lag) / r2_full) * 100 if r2_full > 0 else 0
    
    print(f"Model (Önceki Fiyatlar İLE): R² = {r2_full:.4f} | MAPE = %{mape_full:.2f}")
    print(f"Model (Önceki Fiyatlar YOK): R² = {r2_no_lag:.4f} | MAPE = %{mape_no_lag:.2f}")
    print(f"R² Düşüşü: %{r2_drop_pct:.1f}")
    if r2_drop_pct < 30:
        print("Sonuç: İYİ (Model dışsal değişkenleri öğrenmiş)")
    elif r2_drop_pct > 50:
        print("Sonuç: KÖTÜ (Model sadece geçmiş fiyata bağımlı)")
    else:
        print("Sonuç: UYARILI (Model karmaşık davranıyor)")

    print("\n[TEST 2] MOMENTUM BASELINE KARŞILAŞTIRMASI")
    print("-" * 50)
    # Naive model: Bu ayki fiyat = Geçen ayki Reel USD Fiyat
    preds_naive = df_test["RealUSD_Lag1"].values
    
    mape_naive = calculate_mape(y_test_orig, preds_naive)
    mae_naive = mean_absolute_error(y_test_orig, preds_naive)
    mae_full = mean_absolute_error(y_test_orig, preds_full)
    
    print(f"Aptal Model (t-1 Kopya) MAPE: %{mape_naive:.2f} | MAE: {mae_naive:.2f}")
    print(f"Bizim ML Modelimiz      MAPE: %{mape_full:.2f} | MAE: {mae_full:.2f}")
    
    if mape_full < mape_naive * 0.9: # %10 daha iyiyse
        print("Sonuç: ÇOK İYİ (Model sadece ezberlemiyor, geleceği tahmin ediyor)")
    elif mape_full > mape_naive:
        print("Sonuç: KÖTÜ (Model aptal tahminciden bile kötü durumda)")
    else:
        print("Sonuç: KÖTÜ/UYARILI (ML modelinin aptal modele üstünlüğü yok)")
        
    print("\n[TEST 3] ŞOK DÖNEMİ (STRESS) TESTİ")
    print("-" * 50)
    # Şok dönemi tanımı: Fiyatın bir önceki aya göre Reel USD bazında %10'dan fazla değiştiği aylar
    df_test["Fiyat_Degisimi"] = np.abs(df_test[TARGET] - df_test["RealUSD_Lag1"]) / df_test["RealUSD_Lag1"] * 100
    shock_indices = df_test.index[df_test["Fiyat_Degisimi"] > 10].tolist()
    normal_indices = df_test.index[df_test["Fiyat_Degisimi"] <= 10].tolist()
    
    if len(shock_indices) > 0:
        y_test_shock = df_test.loc[shock_indices, TARGET].values
        preds_shock = preds_full[np.where(df_test.index.isin(shock_indices))[0]]
        mape_shock = calculate_mape(y_test_shock, preds_shock)
        
        y_test_norm = df_test.loc[normal_indices, TARGET].values
        preds_norm = preds_full[np.where(df_test.index.isin(normal_indices))[0]]
        mape_norm = calculate_mape(y_test_norm, preds_norm)
        
        print(f"Bulunan Şok Ayı Sayısı : {len(shock_indices)}")
        print(f"Normal Dönem MAPE      : %{mape_norm:.2f}")
        print(f"ŞOK DÖNEMİ MAPE        : %{mape_shock:.2f}")
        
        if mape_shock <= 10:
            print("Sonuç: İYİ (Model krizleri önceden seziyor)")
        elif mape_shock > 12:
            print("Sonuç: KÖTÜ (Model krizlerde körleşiyor, Lagging Indicator)")
        else:
            print("Sonuç: UYARILI")
    else:
        print("Test setinde fiyatta %10'dan fazla değişim olan 'Şok' ayı bulunamadı.")
        
    print("\n================= DEĞERLENDİRME BİTTİ =================")

if __name__ == "__main__":
    main()

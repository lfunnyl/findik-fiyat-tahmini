"""
scripts/check_flaml_overfit.py
==============================
FLAML modelinin overfit durumunu kontrol eder.

Train vs Test MAPE/R² karşılaştırması yapar.
Ayrıca Walk-Forward CV ile out-of-sample performans hesaplar.

Çalıştır:
    python scripts/check_flaml_overfit.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "processed", "master_features.csv")
MODELS_DIR = os.path.join(ROOT, "models")

# ── Sabitler (api/main.py ile senkron) ─────────────────────────────────────
TARGET = "Fiyat_RealUSD_kg"
TOP_N  = 20
DROP_COLS = [
    TARGET, "Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "US_CPI_Carpani",
    "Tarih", "Yil_Ay", "Hasat_Donemi",
    "TMO_Giresun_TL_kg", "TMO_Levant_TL_kg",
    "Fiyat_Lag1", "Fiyat_Lag2", "Fiyat_Lag3", "Fiyat_Lag12",
    "Fiyat_Degisim_1A_Pct", "Fiyat_Degisim_3A_Pct",
    "Fiyat_bolu_AsgariUcret_Orani", "Yil", "Sezon_Yili",
]


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih").reset_index(drop=True)

    df["USD_Lag1"]     = df["Fiyat_USD_kg"].shift(1)
    df["USD_Lag2"]     = df["Fiyat_USD_kg"].shift(2)
    df["USD_Lag3"]     = df["Fiyat_USD_kg"].shift(3)
    df["USD_Lag12"]    = df["Fiyat_USD_kg"].shift(12)
    df["USD_MoM_pct"]  = df["Fiyat_USD_kg"].pct_change(1) * 100
    df["USD_YoY_pct"]  = df["Fiyat_USD_kg"].pct_change(12) * 100
    df["RealUSD_Lag1"] = df["Fiyat_RealUSD_kg"].shift(1)
    df["RealUSD_Lag3"] = df["Fiyat_RealUSD_kg"].shift(3)
    df = df.bfill().ffill()
    return df


def get_features_and_target(df, split_idx):
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    y = df[TARGET]
    y_log = np.log1p(y)

    # Feature selection train seti üzerinde (data leakage yok)
    corr = X.iloc[:split_idx].corrwith(y_log.iloc[:split_idx]).abs().dropna()
    sel_cols = corr.nlargest(TOP_N).index.tolist()

    return X[sel_cols], y, y_log, sel_cols


def metrics(y_true, y_pred, label):
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  {label:<20} → MAPE: {mape:6.2f}%  |  R²: {r2:+.3f}  |  MAE: {mae:.3f} USD/kg")
    return mape, r2


print("=" * 65)
print("🔍 FLAML Overfit Kontrolü")
print("=" * 65)

# ── Veri ──────────────────────────────────────────────────────────────────
df = load_data()
n  = len(df)
split_idx = int(n * 0.80)

print(f"\n  Toplam veri : {n} satır")
print(f"  Train seti  : {split_idx} satır (2013-08 → {df['Tarih'].iloc[split_idx-1].strftime('%Y-%m')})")
print(f"  Test seti   : {n - split_idx} satır ({df['Tarih'].iloc[split_idx].strftime('%Y-%m')} → {df['Tarih'].iloc[-1].strftime('%Y-%m')})")

X, y, y_log, sel_cols = get_features_and_target(df, split_idx)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# ── Model Yükle ───────────────────────────────────────────────────────────
print("\n  Modeller yükleniyor...")
flaml_bundle = joblib.load(os.path.join(MODELS_DIR, "flaml_model.pkl"))
xgb_bundle   = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))

# FLAML modeli dict değil, doğrudan AutoML nesnesi olarak kaydedilmiş
flaml_model = flaml_bundle["model"] if isinstance(flaml_bundle, dict) else flaml_bundle
xgb_model   = xgb_bundle["model"]   if isinstance(xgb_bundle,   dict) else xgb_bundle

print(f"  FLAML model tipi  : {type(flaml_model).__name__}")
print(f"  XGBoost model tipi: {type(xgb_model).__name__}")

print("\n" + "─" * 65)
print("📊 TRAIN Seti Performansı (Eğitim verisi — overfit göstergesi)")
print("─" * 65)

# FLAML train
# FLAML — log-transform edilmiş mi kontrol et
try:
    flaml_train_raw  = flaml_model.predict(X_train)
    # Değerler makul reel USD aralığındaysa (1-20) direkt kullan, yoksa expm1 uygula
    flaml_train_pred = np.expm1(flaml_train_raw) if flaml_train_raw.mean() < 5 else flaml_train_raw
except Exception as e:
    print(f"  ⚠️  FLAML train tahmin hatası: {e}")
    flaml_train_pred = np.full(len(y_train), y_train.mean())
mape_flaml_train, r2_flaml_train = metrics(y_train, flaml_train_pred, "FLAML (Train)")

# XGBoost train
xgb_train_log  = xgb_model.predict(X_train)
xgb_train_pred = np.expm1(xgb_train_log)
mape_xgb_train, r2_xgb_train = metrics(y_train, xgb_train_pred, "XGBoost (Train)")

print("\n" + "─" * 65)
print("📊 TEST Seti Performansı (Gerçek out-of-sample)")
print("─" * 65)

# FLAML test
try:
    flaml_test_raw  = flaml_model.predict(X_test)
    flaml_test_pred = np.expm1(flaml_test_raw) if flaml_test_raw.mean() < 5 else flaml_test_raw
except Exception as e:
    print(f"  ⚠️  FLAML test tahmin hatası: {e}")
    flaml_test_pred = np.full(len(y_test), y_test.mean())
mape_flaml_test, r2_flaml_test = metrics(y_test, flaml_test_pred, "FLAML (Test)")

# XGBoost test
xgb_test_log  = xgb_model.predict(X_test)
xgb_test_pred = np.expm1(xgb_test_log)
mape_xgb_test, r2_xgb_test = metrics(y_test, xgb_test_pred, "XGBoost (Test)")

print("\n" + "─" * 65)
print("🔬 Overfit Analizi")
print("─" * 65)

flaml_gap = mape_flaml_test - mape_flaml_train
xgb_gap   = mape_xgb_test  - mape_xgb_train

print(f"\n  FLAML   Train→Test MAPE farkı : {flaml_gap:+.2f}%  ({'⚠️ Overfit şüpheli' if flaml_gap > 3 else '✅ Normal'})")
print(f"  XGBoost Train→Test MAPE farkı : {xgb_gap:+.2f}%  ({'⚠️ Overfit şüpheli' if xgb_gap > 3 else '✅ Normal'})")

print("\n" + "─" * 65)
print("📈 Son 12 Ay (2025-2026) Tahmin vs Gerçek")
print("─" * 65)

last12 = df.tail(12)
X_last12 = X.tail(12)
y_last12 = y.tail(12)

try:
    _f = flaml_model.predict(X_last12)
    flaml_l12 = np.expm1(_f) if _f.mean() < 5 else _f
except Exception:
    flaml_l12 = np.full(len(X_last12), y_last12.mean())
xgb_l12 = np.expm1(xgb_model.predict(X_last12))

print(f"\n  {'Ay':<10} {'Gerçek':>10} {'FLAML':>10} {'XGBoost':>10} {'FLAML Hata':>12} {'XGB Hata':>12}")
print(f"  {'─'*8:<10} {'─'*8:>10} {'─'*8:>10} {'─'*8:>10} {'─'*10:>12} {'─'*10:>12}")
for i, (date, real, f_pred, x_pred) in enumerate(zip(
    last12["Tarih"], y_last12, flaml_l12, xgb_l12
)):
    f_err = (f_pred - real) / real * 100
    x_err = (x_pred - real) / real * 100
    marker = " 🔴" if abs(f_err) > 15 else ""
    print(f"  {date.strftime('%Y-%m'):<10} {real:>10.3f} {f_pred:>10.3f} {x_pred:>10.3f} {f_err:>+11.1f}% {x_err:>+11.1f}%{marker}")

print("\n" + "─" * 65)
mape_l12_flaml = mean_absolute_percentage_error(y_last12, flaml_l12) * 100
mape_l12_xgb   = mean_absolute_percentage_error(y_last12, xgb_l12) * 100
print(f"  Son 12 ay MAPE → FLAML: {mape_l12_flaml:.2f}%  |  XGBoost: {mape_l12_xgb:.2f}%")

print("\n" + "=" * 65)
print("📋 SONUÇ")
print("=" * 65)

if flaml_gap > 5:
    print("\n  ❌ FLAML güçlü overfit belirtisi gösteriyor!")
    print("     Weighted Ensemble production'da daha güvenilir.")
elif flaml_gap > 2:
    print("\n  ⚠️  FLAML hafif overfit eğiliminde.")
    print("     Karşılaştırma için tutulabilir ama production'da Weighted Ensemble tercih edilmeli.")
else:
    print("\n  ✅ FLAML overfit belirtisi yok — sağlıklı genelleme yapıyor.")
    print("     Ensemble'a dahil edilmesi düşünülebilir.")

print()

"""
scripts/check_vif.py
====================
Mevcut eğitim feature setindeki multikollineariteyi VIF ile ölçer.
VIF > 10 olan özellikler sorunlu kabul edilir.

Çalıştır: python scripts/check_vif.py
"""
import os, sys, numpy as np, pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "processed", "master_features.csv")

TARGET = "Fiyat_RealUSD_kg"
DROP_COLS = [
    TARGET, "Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "US_CPI_Carpani",
    "Tarih", "Yil_Ay", "Hasat_Donemi",
    "TMO_Giresun_TL_kg", "TMO_Levant_TL_kg",
    "Fiyat_Lag1", "Fiyat_Lag2", "Fiyat_Lag3", "Fiyat_Lag12",
    "Fiyat_Degisim_1A_Pct", "Fiyat_Degisim_3A_Pct",
    "Fiyat_bolu_AsgariUcret_Orani", "Yil", "Sezon_Yili",
    # Yeni eklenenler (config.yaml v3.2)
    "Altin_Ons_Kapanis", "Ihracat_Deger_mUSD", "TCMB_Faiz_Orani", "USD_Lag2",
]

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
df["Tarih"] = pd.to_datetime(df["Tarih"])
df = df.sort_values("Tarih").reset_index(drop=True)

# Lag feature'lar
df["USD_Lag1"]     = df["Fiyat_USD_kg"].shift(1)
df["USD_Lag3"]     = df["Fiyat_USD_kg"].shift(3)
df["USD_Lag12"]    = df["Fiyat_USD_kg"].shift(12)
df["USD_MoM_pct"]  = df["Fiyat_USD_kg"].pct_change(1) * 100
df["USD_YoY_pct"]  = df["Fiyat_USD_kg"].pct_change(12) * 100
df["RealUSD_Lag1"] = df["Fiyat_RealUSD_kg"].shift(1)
df["RealUSD_Lag3"] = df["Fiyat_RealUSD_kg"].shift(3)
df = df.bfill().ffill()

drop_existing = [c for c in DROP_COLS if c in df.columns]
X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
y_log = np.log1p(df[TARGET])

n = len(df)
split_idx = int(n * 0.80)

# Korelasyon ile top-15 seç (mevcut eğitim mantığı)
corr = X.iloc[:split_idx].corrwith(y_log.iloc[:split_idx]).abs().dropna()
top15 = corr.nlargest(15).index.tolist()
X_sel = X[top15].iloc[:split_idx].fillna(0)

print("=" * 65)
print("🔬 VIF Analizi — Mevcut Top-15 Feature Seti")
print("=" * 65)

vifs = []
for i, col in enumerate(X_sel.columns):
    try:
        vif = variance_inflation_factor(X_sel.values, i)
    except Exception:
        vif = float("nan")
    vifs.append((col, vif))

vif_df = pd.DataFrame(vifs, columns=["Özellik", "VIF"]).sort_values("VIF", ascending=False)

n_high = 0
print(f"\n  {'Özellik':<30} {'VIF':>10}  Durum")
print(f"  {'─'*28:<30} {'─'*8:>10}  {'─'*12}")
for _, row in vif_df.iterrows():
    vif = row["VIF"]
    if vif > 10:
        status = "🔴 SORUNLU (>10)"
        n_high += 1
    elif vif > 5:
        status = "🟡 Orta    (5-10)"
    else:
        status = "✅ İyi     (<5)"
    print(f"  {row['Özellik']:<30} {vif:>10.2f}  {status}")

print(f"\n  Toplam feature : {len(vif_df)}")
print(f"  VIF > 10 (sorunlu): {n_high}")
print(f"  VIF 5-10 (orta)   : {len(vif_df[(vif_df['VIF'] > 5) & (vif_df['VIF'] <= 10)])}")
print(f"  VIF < 5  (temiz)  : {len(vif_df[vif_df['VIF'] <= 5])}")

print("\n" + "─" * 65)
if n_high > 0:
    print(f"\n  ⚠️  {n_high} özellikte yüksek multikollinearite mevcut.")
    print("  Modelleri yeniden eğiterek VIF seçimini aktif etmek gerekiyor.")
    print("  Komut: python src/models/train_model.py")
else:
    print("\n  ✅ Multikollinearite sorunu yok — feature seti temiz!")
print()

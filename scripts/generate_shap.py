"""
scripts/generate_shap.py
========================
XGBoost modeli için SHAP değerlerini hesaplar ve görsellerini üretir.

Üretilen görseller → reports/figures/
  - shap_summary_beeswarm.png   : Her özelliğin etkisini gösteren ana grafik
  - shap_feature_importance.png : Bar grafiği — ortalama |SHAP|
  - shap_waterfall_last.png     : Son tahmin için waterfall (tek satır açıklama)
  - shap_dependence_top3.png    : En etkili 3 özellik için bağımlılık grafikleri

Çalıştır:
    python scripts/generate_shap.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # headless — GUI gerektirmez
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# ─── Yollar ──────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "processed", "master_features.csv")
MODELS_DIR = os.path.join(ROOT, "models")
FIG_DIR    = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ─── Sabitler ─────────────────────────────────────────────────────────────────
TARGET    = "Fiyat_RealUSD_kg"
TOP_N     = 20
DROP_COLS = [
    TARGET, "Serbest_Piyasa_TL_kg", "Fiyat_USD_kg", "US_CPI_Carpani",
    "Tarih", "Yil_Ay", "Hasat_Donemi",
    "TMO_Giresun_TL_kg", "TMO_Levant_TL_kg",
    "Fiyat_Lag1", "Fiyat_Lag2", "Fiyat_Lag3", "Fiyat_Lag12",
    "Fiyat_Degisim_1A_Pct", "Fiyat_Degisim_3A_Pct",
    "Fiyat_bolu_AsgariUcret_Orani", "Yil", "Sezon_Yili",
]

# Türkçe özellik isimleri (okunabilirlik için)
FEATURE_LABELS = {
    "RealUSD_Lag1":       "Önceki Ay Reel Fiyat",
    "RealUSD_Lag3":       "3 Ay Öncesi Reel Fiyat",
    "USD_Lag1":           "Önceki Ay USD Fiyat",
    "USD_Lag3":           "3 Ay Öncesi USD Fiyat",
    "USD_MoM_pct":        "Aylık Değişim (%)",
    "USD_YoY_pct":        "Yıllık Değişim (%)",
    "TUFE_Yillik":        "TÜFE (Yıllık %)",
    "USD_TRY_Kur":        "Döviz Kuru (TL/USD)",
    "Uretim_Bin_Ton":     "Üretim (Bin Ton)",
    "Ihracat_Ton":        "İhracat (Ton)",
    "Kuresel_Talep_Idx":  "Küresel Talep İndeksi",
    "Ay_Sin":             "Mevsimsellik (Sin)",
    "Ay_Cos":             "Mevsimsellik (Cos)",
    "Don_Riski_Gun":      "Don Riski (Gün)",
    "Yagis_mm":           "Yağış (mm)",
    "Asgari_Ucret_TL":    "Asgari Ücret (TL)",
    "Benzin_TL_L":        "Yakıt Fiyatı",
    "Secim_Gundemi":      "Seçim Gündemi",
    "Ramazan_Etkisi":     "Ramazan Etkisi",
    "Hasat_Yili_Bitis":   "Hasat Yılı Sonu",
}

# ─── Stil Ayarları ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "axes.edgecolor":   "#334155",
    "axes.labelcolor":  "#e2e8f0",
    "text.color":       "#e2e8f0",
    "xtick.color":      "#94a3b8",
    "ytick.color":      "#94a3b8",
    "grid.color":       "#334155",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

ACCENT  = "#a78bfa"   # mor
ACCENT2 = "#38bdf8"   # mavi
GREEN   = "#4ade80"
RED     = "#f87171"


def load_data_and_model():
    """Veri ve XGBoost modelini yükle, özellik seçimi yap."""
    print("📂 Veri yükleniyor...")
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["Tarih"] = pd.to_datetime(df["Tarih"])
    df = df.sort_values("Tarih").reset_index(drop=True)

    # Lag özellikler (train_model.py ile BİREBİR AYNI olmalı)
    # DİKKAT: Sızıntıyı önlemek için önce .shift(1) alınıp sonra hesaplanır
    df["USD_MoM_pct"]     = df["Fiyat_USD_kg"].shift(1).pct_change(1) * 100
    df["USD_YoY_pct"]     = df["Fiyat_USD_kg"].shift(1).pct_change(12) * 100
    df["RealUSD_MoM_pct"] = df["Fiyat_RealUSD_kg"].shift(1).pct_change(1) * 100
    df["RealUSD_YoY_pct"] = df["Fiyat_RealUSD_kg"].shift(1).pct_change(12) * 100
    
    # Yeni Causal / Momentum Features
    df['RealUSD_MA3'] = df['Fiyat_RealUSD_kg'].shift(1).rolling(window=3).mean()
    df['RealUSD_MA6'] = df['Fiyat_RealUSD_kg'].shift(1).rolling(window=6).mean()
    df['Fiyat_MA3_Farki_Pct'] = (df['Fiyat_RealUSD_kg'].shift(1) - df['RealUSD_MA3']) / df['RealUSD_MA3'] * 100
    df["Kur_Aylik_Ivme"]  = df["USD_TRY_Kapanis"].shift(1).pct_change(1) * 100
    df['Kur_Volatilite_3Ay'] = df['USD_TRY_Kapanis'].shift(1).rolling(window=3).std()
    
    df = df.bfill().ffill()
    
    # --- YENİ EKLENEN: Regime Detection (Şok Alarmı) ---
    volatilite_mean = df['Kur_Volatilite_3Ay'].mean()
    is_shock = (df['Kur_Volatilite_3Ay'] > volatilite_mean * 2)
    if 'Kritik_Don' in df.columns:
        is_shock = is_shock | (df['Kritik_Don'] > 0)
    df['Regime_Shock_Warning'] = np.where(is_shock, 1, 0)
    
    # --- YENİ EKLENEN: TMO Müdahalesi (Policy Causal Feature) ---
    df['TMO_Fiyat_Artis_Pct'] = df['TMO_Giresun_TL_kg'].pct_change(1) * 100
    df['TMO_Mevcut_Makas_Pct'] = (df['TMO_Giresun_TL_kg'] - df['Serbest_Piyasa_TL_kg'].shift(1)) / df['Serbest_Piyasa_TL_kg'].shift(1) * 100

    # Özellik matrisi (Yasaklı lagları çıkar)
    yasakli_laglar = [
        "Fiyat_Lag1", "Fiyat_Lag2", "Fiyat_Lag3", "Fiyat_Lag12",
        "USD_Lag1", "USD_Lag2", "USD_Lag3", "USD_Lag12",
        "RealUSD_Lag1", "RealUSD_Lag3"
    ]
    drop_existing = [c for c in DROP_COLS + yasakli_laglar if c in df.columns]
    X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    
    # SHAP script expects y_log = log(Target)
    y_log = np.log1p(df[TARGET])

    # Modeli yükle
    print("🤖 XGBoost modeli yükleniyor...")
    bundle    = joblib.load(os.path.join(MODELS_DIR, "xgboost_model.pkl"))
    xgb_model = bundle["model"] if isinstance(bundle, dict) else bundle
    sel_cols  = bundle["features"]

    X_sel = X[sel_cols]

    # İnsan okunabilir özellik isimleri
    display_names = [FEATURE_LABELS.get(c, c) for c in sel_cols]

    return X_sel, y_log, xgb_model, sel_cols, display_names, df["Tarih"]


def compute_shap(model, X):
    """TreeExplainer ile SHAP değerlerini hesapla."""
    print("🔢 SHAP değerleri hesaplanıyor...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    expected    = explainer.expected_value
    return shap_values, expected


def plot_beeswarm(shap_values, X, display_names):
    """SHAP beeswarm (summary) grafiği."""
    print("🎨 Beeswarm grafiği oluşturuluyor...")
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.sca(ax)

    X_display = X.copy()
    X_display.columns = display_names

    shap.summary_plot(
        shap_values,
        X_display,
        plot_type="dot",
        max_display=15,
        show=False,
        color_bar_label="Özellik Değeri",
    )

    plt.title("🌰 Fındık Fiyat Modeli — SHAP Özellik Etkileri",
              fontsize=14, fontweight="bold", pad=15, color="#e2e8f0")
    plt.xlabel("SHAP Değeri (Log-Reel USD Tahminine Etkisi)", color="#94a3b8")
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "shap_summary_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  ✅ {path}")


def plot_bar_importance(shap_values, display_names):
    """SHAP bar importance grafiği."""
    print("📊 Bar importance grafiği oluşturuluyor...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[-15:]  # top 15

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(
        [display_names[i] for i in sorted_idx],
        mean_abs[sorted_idx],
        color=ACCENT,
        alpha=0.85,
        edgecolor="#7c3aed",
        linewidth=0.5,
    )

    # Renk degradesi — daha büyük = daha parlak
    for bar, val in zip(bars, mean_abs[sorted_idx]):
        alpha = 0.5 + 0.5 * (val / mean_abs[sorted_idx].max())
        bar.set_alpha(alpha)

    ax.set_xlabel("Ortalama |SHAP Değeri|", color="#94a3b8")
    ax.set_title("Özellik Önem Sıralaması (SHAP)", fontsize=13,
                 fontweight="bold", pad=12, color="#e2e8f0")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "shap_feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  ✅ {path}")


def plot_waterfall(shap_values, expected_value, X, display_names, tarihler):
    """Son satır için waterfall grafiği."""
    print("🌊 Waterfall grafiği oluşturuluyor...")
    last_idx   = -1
    last_shap  = shap_values[last_idx]
    last_date  = tarihler.iloc[last_idx].strftime("%Y-%m")

    # En etkili 10 özelliği göster
    top_idx    = np.argsort(np.abs(last_shap))[-10:][::-1]
    names_top  = [display_names[i] for i in top_idx]
    vals_top   = last_shap[top_idx]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = [GREEN if v > 0 else RED for v in vals_top]
    ax.barh(names_top[::-1], vals_top[::-1], color=colors[::-1],
            alpha=0.85, edgecolor="#1e293b", linewidth=0.5)

    ax.axvline(0, color="#64748b", linewidth=1)
    ax.set_xlabel("SHAP Değeri", color="#94a3b8")
    ax.set_title(f"Tahmin Açıklaması — {last_date} (Son Gözlem)",
                 fontsize=13, fontweight="bold", pad=12, color="#e2e8f0")

    # Baz değer notu
    ax.text(0.01, 0.02, f"Baz değer: {np.expm1(expected_value):.3f} USD/kg",
            transform=ax.transAxes, color="#94a3b8", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "shap_waterfall_last.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  ✅ {path}")


def plot_dependence(shap_values, X, display_names):
    """En etkili 3 özellik için dependence grafikleri."""
    print("📈 Dependence grafikleri oluşturuluyor...")
    mean_abs = np.abs(shap_values).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[-3:][::-1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, feat_idx in zip(axes, top3_idx):
        name    = display_names[feat_idx]
        x_vals  = X.iloc[:, feat_idx].values
        s_vals  = shap_values[:, feat_idx]

        sc = ax.scatter(x_vals, s_vals, c=s_vals, cmap="RdYlGn",
                        alpha=0.7, s=30, edgecolors="none")
        plt.colorbar(sc, ax=ax, label="SHAP")
        ax.axhline(0, color="#64748b", linewidth=1, linestyle="--")
        ax.set_xlabel(name, color="#94a3b8", fontsize=10)
        ax.set_ylabel("SHAP Değeri", color="#94a3b8", fontsize=10)
        ax.set_title(name, fontsize=11, fontweight="bold", color="#e2e8f0")

    fig.suptitle("Özellik Bağımlılık Grafikleri (Top 3)",
                 fontsize=13, fontweight="bold", color="#e2e8f0", y=1.02)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "shap_dependence_top3.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"  ✅ {path}")


def save_shap_json(shap_values, display_names):
    """API için SHAP önem skorlarını JSON olarak kaydet."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    total_shap = mean_abs.sum()
    sorted_idx = np.argsort(mean_abs)[::-1]

    data = {
        "features": [
            {
                "rank":      int(i + 1),
                "name":      display_names[idx],
                "raw_name":  display_names[idx],
                "importance": round(float(mean_abs[idx] / total_shap) * 100, 2),
            }
            for i, idx in enumerate(sorted_idx[:15])
        ],
        "generated_at": pd.Timestamp.now().isoformat(),
    }

    path = os.path.join(ROOT, "models", "shap_importance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✅ {path}")


# ─── Ana Akış ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🌰 SHAP Görsel Üretimi Başlıyor")
    print("=" * 60)

    # SHAP kurulu mu kontrol et
    try:
        import shap
    except ImportError:
        print("❌ SHAP kütüphanesi eksik. Kur: pip install shap")
        sys.exit(1)

    X, y_log, model, sel_cols, display_names, tarihler = load_data_and_model()
    shap_values, expected_value = compute_shap(model, X)

    print("\n🎨 Görseller üretiliyor...")
    plot_beeswarm(shap_values, X, display_names)
    plot_bar_importance(shap_values, display_names)
    plot_waterfall(shap_values, expected_value, X, display_names, tarihler)
    plot_dependence(shap_values, X, display_names)
    save_shap_json(shap_values, display_names)

    print("\n" + "=" * 60)
    print(f"✅ Tüm görseller hazır → reports/figures/")
    print(f"   • shap_summary_beeswarm.png")
    print(f"   • shap_feature_importance.png")
    print(f"   • shap_waterfall_last.png")
    print(f"   • shap_dependence_top3.png")
    print(f"   • models/shap_importance.json  (API için)")
    print("=" * 60)

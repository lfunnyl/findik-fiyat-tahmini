"""
tmo_model.py
============
TMO (Toprak Mahsulleri Ofisi) Fındık Taban Fiyatı Tahmin Modeli

Mantık:
  - TMO her yıl Temmuz/Ağustos'ta taban fiyat açıklar
  - 2013-2025 arası 13 gözlem noktası (yıllık)
  - Küçük örneklem → Ridge Regresyon + Leave-One-Out CV
  - Log-transform ile heterojenlik giderilir
  - Özellikler: USD/TRY, Asgari Ücret, TÜFE, Türkiye üretimi, dünya tüketimi

Çalıştırma:
  python src/models/tmo_model.py
"""

import os
import json
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─── Veri Hazırlama ───────────────────────────────────────────────────────────

def build_tmo_dataset():
    """
    Ağustos ayı TMO açıklamalarını ve o anki makro değişkenleri derler.
    Her satır = bir takvim yılı (2013-2025).
    """
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])

    # Ağustos = TMO açıklama ayı
    ag = df[df['Tarih'].dt.month == 8].copy()
    ag['Yil'] = ag['Tarih'].dt.year

    # Geçen yılın TMO fiyatı (lag)
    ag = ag.sort_values('Yil').reset_index(drop=True)
    ag['TMO_Lag1']         = ag['TMO_Giresun_TL_kg'].shift(1)
    ag['TMO_Buyume_Pct']   = ag['TMO_Giresun_TL_kg'].pct_change() * 100

    # Özellik sütunları
    feature_cols = [
        'USD_TRY_Kapanis',       # Döviz kuru → maliyet baskısı
        'Asgari_Ucret_TL',       # İşçilik maliyeti → devlet referansı
        'TUFE_Yillik_Pct',       # Yıllık enflasyon → satın alma gücü
        'Uretim_Ton_Turkiye',    # Türkiye rekoltesi → arz
        'Dunya_Tuketim_Ton',     # Dünya talebi
        'TMO_Lag1',              # Geçen yılın taban → referans
        'Serbest_Piyasa_TL_kg',  # Piyasa sinyali (Temmuz)
    ]

    avail = [c for c in feature_cols if c in ag.columns]
    tmo_df = ag[['Yil', 'Tarih', 'TMO_Giresun_TL_kg', 'TMO_Buyume_Pct'] + avail].dropna(
        subset=['TMO_Giresun_TL_kg', 'TMO_Lag1']
    )

    logger.info(f"TMO dataset: {len(tmo_df)} yil (2013-2025), {len(avail)} ozellik")
    return tmo_df, avail


# ─── Model Eğitimi (LOO-CV) ───────────────────────────────────────────────────

def train_tmo_model(tmo_df, feature_cols):
    """
    Leave-One-Out Cross-Validation ile Ridge Regresyon.
    Küçük örneklem (13 gözlem) için en uygun CV stratejisi.
    """
    X = tmo_df[feature_cols].values
    y = np.log(tmo_df['TMO_Giresun_TL_kg'].values)   # Log-transform

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    # Alpha seçimi: geniş aralıkta RidgeCV
    alphas = np.logspace(-2, 4, 50)
    ridge_cv = RidgeCV(alphas=alphas, cv=LeaveOneOut())
    ridge_cv.fit(X_s, y)

    logger.info(f"Optimal Ridge Alpha: {ridge_cv.alpha_:.4f}")

    # LOO CV metrikleri
    loo    = LeaveOneOut()
    oof    = np.zeros(len(y))
    for tr_idx, val_idx in loo.split(X_s):
        m = Ridge(alpha=ridge_cv.alpha_)
        m.fit(X_s[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict(X_s[val_idx])

    oof_orig = np.exp(oof)
    y_orig   = np.exp(y)
    mape_loo = np.mean(np.abs((y_orig - oof_orig) / y_orig)) * 100
    mae_loo  = mean_absolute_error(y_orig, oof_orig)
    r2_loo   = r2_score(y_orig, oof_orig)

    logger.info(f"LOO-CV MAPE : {mape_loo:.2f}%")
    logger.info(f"LOO-CV MAE  : {mae_loo:.2f} TL/kg")
    logger.info(f"LOO-CV R2   : {r2_loo:.4f}")

    # Son model: tüm veriyle eğit
    final_model = Ridge(alpha=ridge_cv.alpha_)
    final_model.fit(X_s, y)

    # Katsayı önemi
    coef_df = pd.DataFrame({
        'Ozellik': feature_cols,
        'Katsayi': final_model.coef_,
        'Abs_Katsa': np.abs(final_model.coef_),
    }).sort_values('Abs_Katsa', ascending=False)
    logger.info("\nKatsayi Onemi:")
    for _, row in coef_df.iterrows():
        logger.info(f"  {row['Ozellik']:<28}: {row['Katsayi']:+.4f}")

    return final_model, scaler, {
        'mape': round(mape_loo, 2),
        'mae':  round(mae_loo, 2),
        'r2':   round(r2_loo, 4),
        'alpha': round(ridge_cv.alpha_, 4),
    }, oof_orig, y_orig, coef_df


# ─── Tahmin ───────────────────────────────────────────────────────────────────

def predict_tmo_2026(model, scaler, feature_cols, tmo_df, override_values=None):
    """
    2026 Temmuz/Agustos TMO taban fiyati tahmini.
    2026 girdi degerleri: son 2 yilin buyume hiziyla ekstrapolasyon.
    override_values: dict -> bazi degerleri manuel gecersiz kilmak icin
    """
    last = tmo_df.iloc[-1]   # 2025 Agustos
    prev = tmo_df.iloc[-2]   # 2024 Agustos

    def growth(col):
        if col in last and col in prev and prev[col] != 0:
            return float(last[col]) / float(prev[col])
        return 1.0

    defaults = {}
    for col in feature_cols:
        if col == 'TMO_Lag1':
            # 2026 modeline girerken, 2025'te aciklanan TMO = gececek yilin TMO'su 200 TL
            defaults[col] = float(last['TMO_Giresun_TL_kg'])
        elif col == 'Uretim_Ton_Turkiye':
            # Uretim miktari gorece stabil; 5 yil ortalamasi
            defaults[col] = float(tmo_df[col].tail(5).mean())
        elif col in last.index:
            # Diger kolonlar: son yilin buyume hiziyla ekstrapolasyon
            defaults[col] = float(last[col]) * growth(col)
        else:
            defaults[col] = 0.0

    if override_values:
        defaults.update(override_values)

    logger.info("2026 icin kullanilan girdi degerleri:")
    for k, v in defaults.items():
        logger.info(f"  {k:<28}: {v:.2f}")

    X_input    = pd.DataFrame([defaults])[feature_cols].values
    X_scaled   = scaler.transform(X_input)
    y_pred_log = model.predict(X_scaled)[0]
    tmo_pred   = np.exp(y_pred_log)

    return tmo_pred, defaults


def bootstrap_tmo_ci(model, scaler, feature_cols, input_dict, n=500, noise_pct=0.05):
    """Bootstrap CI: girdi değerlerine ±%5 gürültü ekle."""
    preds = []
    base = np.array([[input_dict[c] for c in feature_cols]])
    rng = np.random.default_rng(42)
    for _ in range(n):
        noisy = base * (1 + rng.normal(0, noise_pct, base.shape))
        noisy = np.clip(noisy, 0, None)
        y_pred = np.exp(model.predict(scaler.transform(noisy))[0])
        preds.append(y_pred)
    return np.percentile(preds, [5, 25, 50, 75, 95])


# ─── Grafik ───────────────────────────────────────────────────────────────────

def plot_tmo_results(tmo_df, oof_pred, coef_df, pred_2026, ci, metrics):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    plt.style.use('dark_background')

    # ── 1. Gerçek vs Tahmin (LOO-CV) ──
    ax1 = axes[0]
    years = tmo_df['Yil'].values
    actual = tmo_df['TMO_Giresun_TL_kg'].values
    ax1.plot(years, actual, 'o-', color='#6fcf97', lw=2.5, markersize=8, label='Gercek TMO')
    ax1.plot(years, oof_pred, 's--', color='#7c6af7', lw=2, markersize=7, label='LOO-CV Tahmin')
    ax1.axhline(y=pred_2026, color='#ff9800', lw=2.5, linestyle=':', label=f'2026 Tahmin: {pred_2026:.0f} TL')
    ax1.fill_between([2026.3, 2026.7], [ci[0]], [ci[4]], alpha=0.3, color='#ff9800', label='%90 CI')
    ax1.set_title('TMO Taban Fiyati: Gercek vs Tahmin\n(LOO-CV Validasyon)', fontsize=11)
    ax1.set_xlabel('Yil')
    ax1.set_ylabel('TL/kg')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.05, 0.95, f'LOO MAPE: %{metrics["mape"]:.1f}\nLOO R2: {metrics["r2"]:.3f}',
             transform=ax1.transAxes, fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1d2e', alpha=0.8, edgecolor='gray'))

    # ── 2. Katsayı Önemi ──
    ax2 = axes[1]
    colors_c = ['#ff6b6b' if c > 0 else '#6fcf97' for c in coef_df['Katsayi']]
    bars = ax2.barh(coef_df['Ozellik'], coef_df['Katsayi'], color=colors_c, edgecolor='white', linewidth=0.5)
    ax2.axvline(0, color='white', lw=0.8, alpha=0.4)
    ax2.set_title('Ridge Katsayilari\n(Pozitif=arttirir, Negatif=azaltir)', fontsize=11)
    ax2.set_xlabel('Katsayi Degeri (standardize)')
    ax2.grid(True, alpha=0.15, axis='x')

    # ── 3. 2026 TMO Tahmin Dağılımı ──
    ax3 = axes[2]
    ax3.barh(['%95 Ust', '%75 Ust', 'Medyan (%50)', '%25 Alt', '%5 Alt'],
             [ci[4], ci[3], ci[2], ci[1], ci[0]],
             color=['#ff6b6b', '#ffb347', '#7c6af7', '#47b8ff', '#6fcf97'],
             edgecolor='white', linewidth=0.5, height=0.5)
    ax3.axvline(pred_2026, color='#ff9800', lw=2.5, linestyle='--', label=f'Nokta Tahmin: {pred_2026:.0f} TL')
    # 2025 referans
    tmo_2025 = tmo_df.iloc[-1]['TMO_Giresun_TL_kg']
    ax3.axvline(tmo_2025, color='#6fcf97', lw=1.5, linestyle=':', alpha=0.7, label=f'2025 Gercek: {tmo_2025:.0f} TL')
    for i, (val, label) in enumerate(zip([ci[4], ci[3], ci[2], ci[1], ci[0]],
                                          ['%95', '%75', 'Med', '%25', '%5'])):
        ax3.text(val + 1, i, f'{val:.0f} TL', va='center', fontsize=9, color='white')
    ax3.set_title(f'2026 TMO Tahmin Dagilimi\n(Bootstrap %90 Guven Araligi)', fontsize=11)
    ax3.set_xlabel('TL/kg')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.15, axis='x')

    plt.suptitle('TMO Taban Fiyati Tahmin Modeli 2026', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fpath = os.path.join(FIGURES_DIR, '11_tmo_model.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight', facecolor='#0f1117')
    plt.close()
    logger.info(f"Grafik kaydedildi: {fpath}")


# ─── Ana ─────────────────────────────────────────────────────────────────────

def main():
    tmo_df, feature_cols = build_tmo_dataset()

    logger.info("\n" + "=" * 55)
    logger.info("TMO TABAN FIYATI TAHMIN MODELI (Ridge LOO-CV)")
    logger.info("=" * 55)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Findik_Fiyat_Modelleri")

    with mlflow.start_run(run_name="TMO_Taban_Modeli"):
        model, scaler, metrics, oof_pred, y_orig, coef_df = train_tmo_model(tmo_df, feature_cols)
        mlflow.log_metrics({"LOO_MAPE": metrics['mape'], "LOO_MAE": metrics['mae'], "LOO_R2": metrics['r2']})

        logger.info("\n2026 TMO Tahmini:")

    pred_2026, input_vals = predict_tmo_2026(model, scaler, feature_cols, tmo_df)
    ci = bootstrap_tmo_ci(model, scaler, feature_cols, input_vals)

    logger.info(f"  Nokta Tahmin : {pred_2026:.1f} TL/kg")
    logger.info(f"  %90 CI       : [{ci[0]:.1f} — {ci[4]:.1f}] TL/kg")
    logger.info(f"  %50 CI       : [{ci[1]:.1f} — {ci[3]:.1f}] TL/kg")
    logger.info(f"  2025 Gercek  : {tmo_df.iloc[-1]['TMO_Giresun_TL_kg']:.1f} TL/kg")
    buyume = (pred_2026 / tmo_df.iloc[-1]['TMO_Giresun_TL_kg'] - 1) * 100
    logger.info(f"  Beklenen Buyume: %{buyume:.1f}")

    plot_tmo_results(tmo_df, oof_pred, coef_df, pred_2026, ci, metrics)

    # Model kaydet
    bundle = {
        'model':        model,
        'scaler':       scaler,
        'feature_cols': feature_cols,
        'metrics':      metrics,
        'tmo_dataset':  tmo_df.to_dict('records'),
    }
    joblib.dump(bundle, os.path.join(MODELS_DIR, 'tmo_model.pkl'))

    summary = {
        'pred_2026':    round(pred_2026, 1),
        'ci_p5':        round(ci[0], 1),
        'ci_p25':       round(ci[1], 1),
        'ci_p50':       round(ci[2], 1),
        'ci_p75':       round(ci[3], 1),
        'ci_p95':       round(ci[4], 1),
        'mape_loo':     metrics['mape'],
        'r2_loo':       metrics['r2'],
        'tmo_2025':     float(tmo_df.iloc[-1]['TMO_Giresun_TL_kg']),
        'buyume_pct':   round(buyume, 1),
    }
    with open(os.path.join(MODELS_DIR, 'tmo_prediction_2026.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("\nModel kaydedildi: models/tmo_model.pkl")
    logger.info("Tahmin JSON: models/tmo_prediction_2026.json")
    return summary


if __name__ == "__main__":
    result = main()
    print(f"\n[SONUC] 2026 TMO Tahmin: {result['pred_2026']:.1f} TL/kg "
          f"[%90 CI: {result['ci_p5']:.1f} - {result['ci_p95']:.1f}]")

"""
residual_analysis.py
====================
Model Hata Analizi (Residual Analysis)

Amaç:
  - Modelin ne zaman ne kadar yanıldığını sistematik olarak analiz etmek
  - Hata otokorelasyonu (Durbin-Watson) ile sistematik hata tespiti
  - Hasat vs non-hasat sezonu hata karşılaştırması
  - Hata dağılımı (normality test + Q-Q plot)
  - En büyük hataların hangi dönemlere denk geldiği

Çıktı:
  - reports/figures/13_residual_analysis.png
  - reports/figures/14_error_by_season.png
  - models/residual_stats.json
"""

import os
import json
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET = 'Fiyat_RealUSD_kg'
DROP_COLS = [
    TARGET, 'Serbest_Piyasa_TL_kg', 'Fiyat_USD_kg', 'US_CPI_Carpani',
    'Tarih', 'Yil_Ay', 'Hasat_Donemi',
    'TMO_Giresun_TL_kg', 'TMO_Levant_TL_kg',
    'Fiyat_Lag1', 'Fiyat_Lag2', 'Fiyat_Lag3', 'Fiyat_Lag12',
    'Fiyat_Degisim_1A_Pct', 'Fiyat_Degisim_3A_Pct',
    'Fiyat_bolu_AsgariUcret_Orani', 'Yil', 'Sezon_Yili',
]


def load_and_predict():
    """Veri yükle, model tahminleri üret, test setini hazırla."""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih').reset_index(drop=True)

    # Feature Engineering
    df['USD_Lag1']     = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']     = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']     = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']    = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']  = df['Fiyat_USD_kg'].pct_change(1) * 100
    df['USD_YoY_pct']  = df['Fiyat_USD_kg'].pct_change(12) * 100
    df['RealUSD_Lag1'] = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3'] = df['Fiyat_RealUSD_kg'].shift(3)
    df = df.bfill().ffill()

    split_idx = int(len(df) * 0.80)
    df_test   = df.iloc[split_idx:].copy()
    df_train  = df.iloc[:split_idx].copy()

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_all = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    y_log_all = np.log1p(df[TARGET])
    y_raw_all = df[TARGET]

    # Feature selection (train setinde)
    X_train = X_all.iloc[:split_idx]
    y_log_train = y_log_all.iloc[:split_idx]
    corr = X_train.corrwith(y_log_train).abs().dropna()
    sel_cols = corr.nlargest(20).index.tolist()

    X_test = X_all.iloc[split_idx:][sel_cols]
    y_test = y_raw_all.iloc[split_idx:]
    dates_test = df_test['Tarih']

    # Model yükle
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.pkl')
    lgb_path  = os.path.join(MODELS_DIR, 'lightgbm_model.pkl')
    w_path    = os.path.join(MODELS_DIR, 'ensemble_weights.json')

    if not os.path.exists(xgb_path):
        logger.error("Modeller bulunamadı. Önce train_model.py çalıştırın.")
        return None

    m_xgb = joblib.load(xgb_path)
    m_lgb = joblib.load(lgb_path)
    with open(w_path, 'r') as f:
        weights = json.load(f)

    feat_xgb = m_xgb['features']
    feat_lgb = m_lgb['features']

    # Sadece ortak feature'ları kullan
    common_xgb = [c for c in feat_xgb if c in X_all.columns]
    common_lgb = [c for c in feat_lgb if c in X_all.columns]

    X_te_xgb = X_all.iloc[split_idx:][common_xgb]
    X_te_lgb = X_all.iloc[split_idx:][common_lgb]

    p_xgb = np.expm1(m_xgb['model'].predict(X_te_xgb))
    p_lgb = np.expm1(m_lgb['model'].predict(X_te_lgb))
    y_pred = weights['XGBoost'] * p_xgb + weights['LightGBM'] * p_lgb + weights['Ridge'] * p_xgb

    return {
        'y_true': y_test.values,
        'y_pred': y_pred,
        'residuals': y_test.values - y_pred,
        'abs_errors': np.abs(y_test.values - y_pred),
        'pct_errors': np.abs((y_test.values - y_pred) / np.where(y_test.values == 0, 1, y_test.values)) * 100,
        'dates': dates_test.values,
        'months': pd.DatetimeIndex(dates_test).month,
        'df_test': df_test.reset_index(drop=True),
    }


def durbin_watson(residuals):
    """Durbin-Watson istatistiği hesapla."""
    diff = np.diff(residuals)
    dw = np.sum(diff ** 2) / np.sum(residuals ** 2)
    return dw


def plot_residual_analysis(data: dict):
    """6-panel residual analiz grafiği."""
    y_true    = data['y_true']
    y_pred    = data['y_pred']
    residuals = data['residuals']
    dates     = pd.to_datetime(data['dates'])
    months    = data['months']
    pct_errs  = data['pct_errors']

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    DARK_BG = '#1a1d2e'
    GRID_C  = 'rgba(255,255,255,0.08)'
    ACCENT  = '#7c6af7'
    GREEN   = '#6fcf97'
    RED     = '#ff6b6b'
    ORANGE  = '#ff9800'
    TEXT_C  = '#cccccc'

    def style_ax(ax, title):
        ax.set_facecolor(DARK_BG)
        ax.set_title(title, color=TEXT_C, fontsize=10, fontweight='bold', pad=8)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        ax.grid(True, alpha=0.15, color='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('rgba(255,255,255,0.1)')

    # ── Panel 1: Gerçek vs Tahmin (Zaman Serisi) ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(dates, y_true, label='Gerçek', color='white', lw=2, zorder=3)
    ax1.plot(dates, y_pred, label='Tahmin (Ensemble)', color=ACCENT,
             lw=2, linestyle='--', zorder=2)
    ax1.fill_between(dates, y_true, y_pred,
                     where=y_pred < y_true, alpha=0.2, color=GREEN, label='Düşük Tahmin')
    ax1.fill_between(dates, y_true, y_pred,
                     where=y_pred >= y_true, alpha=0.2, color=RED, label='Yüksek Tahmin')
    ax1.set_ylabel('Reel USD/kg', color=TEXT_C, fontsize=9)
    ax1.legend(fontsize=8, facecolor=DARK_BG, edgecolor='gray', labelcolor=TEXT_C, loc='upper left')
    style_ax(ax1, '📈 Gerçek vs Tahmin — Test Seti (Reel USD/kg)')

    # ── Panel 2: Kümülatif Hata (Birikim) ─────────────────────────────────────
    ax_cum = fig.add_subplot(gs[0, 2])
    cumulative_err = np.cumsum(residuals)
    colors_cum = [GREEN if c > 0 else RED for c in cumulative_err]
    ax_cum.bar(range(len(cumulative_err)), cumulative_err, color=colors_cum, alpha=0.8, width=0.9)
    ax_cum.axhline(0, color='white', lw=1, linestyle='--')
    ax_cum.set_xlabel('Test Gözlem Sırası', color=TEXT_C, fontsize=8)
    ax_cum.set_ylabel('Kümülatif Hata (USD)', color=TEXT_C, fontsize=8)
    style_ax(ax_cum, '📉 Kümülatif Hata (Bias Testi)')

    # ── Panel 3: Residual Scatter (Tahmin vs Hata) ───────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    sc = ax2.scatter(y_pred, residuals, c=pct_errs, cmap='RdYlGn_r',
                     alpha=0.8, s=60, vmin=0, vmax=20)
    ax2.axhline(0, color='white', lw=1.5, linestyle='--')
    plt.colorbar(sc, ax=ax2, label='MAPE (%)', fraction=0.04)
    ax2.set_xlabel('Tahmin (USD/kg)', color=TEXT_C, fontsize=8)
    ax2.set_ylabel('Residual', color=TEXT_C, fontsize=8)
    style_ax(ax2, '🔵 Residual vs Tahmin')

    # ── Panel 4: Hata Histogramı + Normal Dağılım ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    n, bins, patches = ax3.hist(residuals, bins=15, color=ACCENT, alpha=0.7, edgecolor='white', lw=0.5)
    # Normal fit
    mu, std = stats.norm.fit(residuals)
    xmin, xmax = ax3.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax3.plot(x, p * len(residuals) * (bins[1] - bins[0]),
             color=ORANGE, lw=2, label=f'Normal: μ={mu:.3f}, σ={std:.3f}')
    ax3.axvline(0, color='white', lw=1.5, linestyle='--')
    ax3.legend(fontsize=7.5, facecolor=DARK_BG, edgecolor='gray', labelcolor=TEXT_C)
    ax3.set_xlabel('Residual (USD/kg)', color=TEXT_C, fontsize=8)
    style_ax(ax3, '📊 Hata Dağılımı (Histogram + Normal Fit)')

    # ── Panel 5: Q-Q Plot ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist='norm')
    ax4.scatter(osm, osr, color=ACCENT, alpha=0.7, s=45)
    line_x = np.array([min(osm), max(osm)])
    ax4.plot(line_x, slope * line_x + intercept, color=ORANGE, lw=2, label=f'R²={r**2:.3f}')
    ax4.set_xlabel('Teorik Nicelikler', color=TEXT_C, fontsize=8)
    ax4.set_ylabel('Gözlem Nicelikler', color=TEXT_C, fontsize=8)
    ax4.legend(fontsize=8, facecolor=DARK_BG, edgecolor='gray', labelcolor=TEXT_C)
    style_ax(ax4, '📐 Q-Q Plot (Normal Uygunluk)')

    # ── Panel 6: Aylara Göre MAPE (Kutu Grafiği) ──────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    month_errors = {m: [] for m in range(1, 13)}
    for err, m in zip(pct_errs, months):
        month_errors[m].append(err)
    tr_months = {1:'Oca', 2:'Şub', 3:'Mar', 4:'Nis', 5:'May', 6:'Haz',
                 7:'Tem', 8:'Ağu', 9:'Eyl', 10:'Eki', 11:'Kas', 12:'Ara'}
    labels = [tr_months[m] for m in range(1, 13)]
    data_box = [month_errors[m] if month_errors[m] else [0] for m in range(1, 13)]
    means = [np.mean(d) for d in data_box]
    bar_colors = ['#ff6b6b' if m in [8, 9, 10] else ACCENT for m in range(1, 13)]
    bars = ax5.bar(labels, means, color=bar_colors, alpha=0.8, edgecolor='white', lw=0.5)
    for bar, val in zip(bars, means):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', color=TEXT_C, fontsize=7.5, fontweight='bold')
    ax5.set_ylabel('Ortalama MAPE (%)', color=TEXT_C, fontsize=8)
    ax5.axhline(np.mean(pct_errs), color=ORANGE, lw=1.5, linestyle='--',
                label=f'Genel Ort: {np.mean(pct_errs):.1f}%')
    ax5.legend(fontsize=8, facecolor=DARK_BG, edgecolor='gray', labelcolor=TEXT_C)
    style_ax(ax5, '📅 Aylara Göre Ortalama MAPE — Hasat (Ağu-Eki) kırmızı')

    # ── Panel 7: Büyük Hata Olayları ──────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    top_n = min(8, len(dates))
    top_idx = np.argsort(pct_errs)[-top_n:][::-1]
    top_dates = [pd.to_datetime(dates[i]).strftime('%Y-%m') for i in top_idx]
    top_vals  = [pct_errs[i] for i in top_idx]
    colors_bar = ['#ff4444' if v > 20 else ORANGE if v > 10 else GREEN for v in top_vals]
    bars2 = ax6.barh(top_dates[::-1], top_vals[::-1], color=colors_bar[::-1], alpha=0.85)
    for bar2, val in zip(bars2, top_vals[::-1]):
        ax6.text(bar2.get_width() + 0.3, bar2.get_y() + bar2.get_height()/2,
                 f'{val:.1f}%', va='center', color=TEXT_C, fontsize=7.5, fontweight='bold')
    ax6.set_xlabel('MAPE (%)', color=TEXT_C, fontsize=8)
    style_ax(ax6, f'⚠️ En Büyük {top_n} Hata Ayı')

    # ── Üst başlık ─────────────────────────────────────────────────────────────
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(pct_errs)
    dw   = durbin_watson(residuals)
    _, p_norm = stats.normaltest(residuals)

    fig.suptitle(
        f'Fındık Fiyat Modeli — Hata Analizi Raporu\n'
        f'MAE: {mae:.3f} USD/kg  |  RMSE: {rmse:.3f}  |  R²: {r2:.4f}  |  '
        f'MAPE: {mape:.2f}%  |  Durbin-Watson: {dw:.3f}  |  '
        f'Normallik p={p_norm:.3f}',
        color='white', fontsize=12, fontweight='bold', y=0.98
    )

    fpath = os.path.join(FIGURES_DIR, '13_residual_analysis.png')
    fig.savefig(fpath, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Hata analizi grafiği → {fpath}")
    return fpath


def compute_stats(data: dict) -> dict:
    """İstatistiksel özetleri hesapla ve JSON olarak kaydet."""
    y_true    = data['y_true']
    y_pred    = data['y_pred']
    residuals = data['residuals']
    pct_errs  = data['pct_errors']
    months    = data['months']

    dw = durbin_watson(residuals)
    _, p_norm = stats.normaltest(residuals)
    skew = stats.skew(residuals)
    kurt = stats.kurtosis(residuals)

    hasat_mask = np.isin(months, [8, 9, 10])
    mape_hasat   = float(np.mean(pct_errs[hasat_mask]))   if hasat_mask.any()  else None
    mape_nonhasat= float(np.mean(pct_errs[~hasat_mask]))  if (~hasat_mask).any() else None

    stats_dict = {
        "n_test_samples": int(len(y_true)),
        "metrics": {
            "MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "R2":   round(float(r2_score(y_true, y_pred)), 4),
            "MAPE": round(float(np.mean(pct_errs)), 2),
        },
        "residual_stats": {
            "mean":   round(float(np.mean(residuals)), 4),
            "std":    round(float(np.std(residuals)), 4),
            "min":    round(float(np.min(residuals)), 4),
            "max":    round(float(np.max(residuals)), 4),
            "skewness": round(float(skew), 4),
            "kurtosis": round(float(kurt), 4),
        },
        "diagnostics": {
            "durbin_watson": round(float(dw), 4),
            "normality_test_pvalue": round(float(p_norm), 4),
            "is_autocorrelated": bool(dw < 1.5 or dw > 2.5),
            "is_normal": bool(p_norm > 0.05),
        },
        "seasonal_breakdown": {
            "mape_hasat_season_pct":   round(mape_hasat, 2)    if mape_hasat    else None,
            "mape_non_hasat_pct":      round(mape_nonhasat, 2) if mape_nonhasat else None,
        }
    }

    out_path = os.path.join(MODELS_DIR, 'residual_stats.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, indent=2, ensure_ascii=False)

    logger.info("=" * 55)
    logger.info("📊 HATA ANALİZİ SONUÇLARI")
    logger.info("=" * 55)
    logger.info(f"  MAE  : {stats_dict['metrics']['MAE']:.4f} USD/kg")
    logger.info(f"  RMSE : {stats_dict['metrics']['RMSE']:.4f}")
    logger.info(f"  R²   : {stats_dict['metrics']['R2']:.4f}")
    logger.info(f"  MAPE : {stats_dict['metrics']['MAPE']:.2f}%")
    logger.info(f"  Durbin-Watson : {dw:.3f}  {'⚠️ Otokorelasyon var!' if bool(dw < 1.5 or dw > 2.5) else '✅ Normal'}")
    logger.info(f"  Normallik p   : {p_norm:.4f}  {'✅ Normal dağılım' if p_norm > 0.05 else '⚠️ Normal değil'}")
    logger.info(f"  Hasat Sezonu MAPE : {mape_hasat:.2f}% | Diğer : {mape_nonhasat:.2f}%")
    logger.info(f"  Sonuçlar kaydedildi: {out_path}")
    logger.info("=" * 55)

    return stats_dict


def run_residual_analysis():
    logger.info("Residual (Hata) Analizi başlatılıyor...")
    data = load_and_predict()
    if data is None:
        return
    stats_dict = compute_stats(data)
    plot_residual_analysis(data)
    logger.info("✅ Hata analizi tamamlandı.")
    return stats_dict


if __name__ == "__main__":
    run_residual_analysis()

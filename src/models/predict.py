"""
predict.py
==========
Fındık Fiyatı Tahmin Projesi - Inference & Senaryo Analizi

Kullanım:
  # Tek tahmin (interaktif)
  python src/models/predict.py

  # Senaryo analizi
  python src/models/predict.py --senaryo

Mantık:
  1. Eğitilmiş XGBoost ve Ridge modelleri yüklenir (Weighted Ensemble: 0.72 + 0.28)
  2. Kullanıcı mevcut ay değişkenlerini girer (veya son ay verisi otomatik alınır)
  3. Reel USD tahmini yapılır → Güncel kur ile TL'ye çevrilir
  4. Güven aralığı (bootstrap) hesaplanır
  5. Senaryo tablosu oluşturulur (USD kuru / rekolte / enerji bazlı)
"""

import os
import json
import logging
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# US CPI tablosu (build_features.py ile senkronize)
US_CPI_TABLE = {
    2013: 69.04, 2014: 70.14, 2015: 70.25, 2016: 71.19,
    2017: 72.67, 2018: 74.38, 2019: 75.71, 2020: 76.34,
    2021: 80.30, 2022: 88.54, 2023: 94.17,
    2024: 100.00, 2025: 102.80, 2026: 105.70,
}
CPI_BAZ_YILI = 2024

TARGET         = 'Fiyat_RealUSD_kg'
TOP_N_FEATURES = 20

DROP_COLS = [
    TARGET, 'Serbest_Piyasa_TL_kg', 'Fiyat_USD_kg', 'US_CPI_Carpani',
    'Tarih', 'Yil_Ay', 'Hasat_Donemi',
    'TMO_Giresun_TL_kg', 'TMO_Levant_TL_kg',
    'Fiyat_Lag1', 'Fiyat_Lag2', 'Fiyat_Lag3', 'Fiyat_Lag12',
    'Fiyat_Degisim_1A_Pct', 'Fiyat_Degisim_3A_Pct',
    'Fiyat_bolu_AsgariUcret_Orani', 'Yil', 'Sezon_Yili',
]


# ─── Model Yükleme ────────────────────────────────────────────────────────────

def load_models():
    """Eğitilmiş XGBoost ve Ridge modellerini yükle."""
    xgb_bundle   = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    xgb_model    = xgb_bundle['model']
    feature_cols = xgb_bundle['features']

    ridge_bundle = joblib.load(os.path.join(MODELS_DIR, 'lightgbm_model.pkl'))
    lgb_model    = ridge_bundle['model']

    # Ensemble ağırlıkları
    with open(os.path.join(MODELS_DIR, 'ensemble_weights.json'), 'r') as f:
        weights = json.load(f)

    logger.info(f"Modeller yüklendi. Özellik sayısı: {len(feature_cols)}")
    logger.info(f"Ensemble ağırlıkları: XGBoost={weights['XGBoost']:.2f}, LightGBM={weights['LightGBM']:.2f}, Ridge={weights['Ridge']:.2f}")
    return xgb_model, lgb_model, feature_cols, weights


# ─── Veri Hazırlama ───────────────────────────────────────────────────────────

def load_history():
    """master_features.csv'yi yükle ve USD lag'larını hesapla."""
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih').reset_index(drop=True)

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


def get_selected_features(df):
    """Eğitimdeki feature seçimini replika et (train seti üzerinde korelasyon)."""
    split_idx = int(len(df) * 0.80)
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    y_log = np.log1p(df[TARGET])
    X_train = X.iloc[:split_idx]
    y_train_log = y_log.iloc[:split_idx]
    corr = X_train.corrwith(y_train_log).abs().dropna()
    sel = corr.nlargest(TOP_N_FEATURES).index.tolist()
    return sel, X


def predict_reel_usd(xgb_model, lgb_model, weights, row_features):
    """
    Tek bir satır (1 aylık veri) için reel USD tahmini yapar.

    Dönüş: {'reel_usd': float, 'xgb_pred': float, 'lgb_pred': float}
    """
    X_arr = pd.DataFrame([row_features])
    pred_xgb_log = xgb_model.predict(X_arr)[0]
    pred_lgb_log = lgb_model.predict(X_arr)[0]

    pred_xgb = np.expm1(pred_xgb_log)
    pred_lgb = np.expm1(pred_lgb_log)

    # Weighted ensemble (Ridge ağırlığı için XGBoost kullanıyoruz - Ridge'in scaler'ı yok burada)
    ensemble = weights['XGBoost'] * pred_xgb + weights['LightGBM'] * pred_lgb + weights['Ridge'] * pred_xgb
    return {
        'reel_usd':  ensemble,
        'xgb_pred':  pred_xgb,
        'lgb_pred':  pred_lgb,
    }


def reel_usd_to_tl(reel_usd, usd_try_kur, tahmin_yili=2026):
    """
    Reel USD → Nominal TL dönüşümü.

    Formül:
      Nominal USD = Reel USD / CPI_Carpani
      TL Fiyat   = Nominal_USD × USD_TRY_Kur
    """
    cpi_carpani  = US_CPI_TABLE[CPI_BAZ_YILI] / US_CPI_TABLE.get(tahmin_yili, US_CPI_TABLE[CPI_BAZ_YILI])
    nominal_usd  = reel_usd / cpi_carpani
    tl_fiyat     = nominal_usd * usd_try_kur
    return nominal_usd, tl_fiyat


# ─── Bootstrap Güven Aralığı ─────────────────────────────────────────────────

def bootstrap_ci(xgb_model, lgb_model, weights, row_features, n_bootstrap=500, noise_pct=0.03):
    """
    Basit parametrik bootstrap ile %90 güven aralığı.
    row_features'a küçük rastgele gürültü ekleyerek tahmin dağılımı oluşturur.
    noise_pct: özelliklere eklenen oran gürültüsü (varsayılan: ±%3)
    """
    predictions = []
    base = np.array([list(row_features.values())])

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        noisy = base * (1 + rng.normal(0, noise_pct, base.shape))
        X_noisy = pd.DataFrame(noisy, columns=list(row_features.keys()))
        res = predict_reel_usd(xgb_model, lgb_model, weights, row_features)
        # Noisy tahmin için doğrudan hesapla
        p_xgb = np.expm1(xgb_model.predict(X_noisy)[0])
        p_lgb = np.expm1(lgb_model.predict(X_noisy)[0])
        p_ens = weights['XGBoost'] * p_xgb + weights['LightGBM'] * p_lgb + weights['Ridge'] * p_xgb
        predictions.append(p_ens)

    return np.percentile(predictions, [5, 50, 95])


# ─── Senaryo Analizi ──────────────────────────────────────────────────────────

def scenario_analysis(xgb_model, lgb_model, weights, base_row, sel_cols,
                      usd_try_kur, tahmin_yili=2026):
    """
    Temel değişkenleri değiştirerek fiyat değişimini simüle eder.
    """
    logger.info("\n" + "═"*60)
    logger.info("📊 SENARYO ANALİZİ — Hassasiyet Testi")
    logger.info("═"*60)

    base_pred   = predict_reel_usd(xgb_model, lgb_model, weights, base_row)['reel_usd']
    _, base_tl  = reel_usd_to_tl(base_pred, usd_try_kur, tahmin_yili)

    senaryolar = []

    # Senaryo 1: USD/TRY kuru değişimi (model içi etki yok — sadece çeviri etkisi)
    for kur_degisim in [-0.10, -0.05, 0.0, +0.05, +0.10, +0.20]:
        yeni_kur  = usd_try_kur * (1 + kur_degisim)
        _, yeni_tl = reel_usd_to_tl(base_pred, yeni_kur, tahmin_yili)
        senaryolar.append({
            'Senaryo': f'USD/TRY kur {kur_degisim:+.0%}',
            'Reel USD/kg': round(base_pred, 3),
            'TL/kg (Tahmin)': round(yeni_tl, 2),
            'TL Fark': round(yeni_tl - base_tl, 2),
            'Fark%': round((yeni_tl - base_tl) / base_tl * 100, 1)
        })

    logger.info("\n  Kur Duyarlılığı (Model + Kur Etkisi):")
    for s in senaryolar:
        logger.info(f"  {s['Senaryo']:<30} → {s['TL/kg (Tahmin)']:>8.2f} TL/kg  ({s['Fark%']:+.1f}%)")

    # Senaryo 2: USD_Lag1 değişimi (geçen ay fiyatı etkisi)
    logger.info("\n  Geçen Ay USD Fiyatı Duyarlılığı:")
    senaryolar2 = []
    if 'USD_Lag1' in base_row:
        base_lag = base_row['USD_Lag1']
        for pct in [-0.15, -0.10, -0.05, 0.0, +0.05, +0.10, +0.15]:
            mod_row = dict(base_row)
            mod_row['USD_Lag1'] = base_lag * (1 + pct)
            if 'USD_Lag2' in mod_row:
                mod_row['RealUSD_Lag1'] = mod_row.get('RealUSD_Lag1', base_lag) * (1 + pct)
            mod_pred = predict_reel_usd(xgb_model, lgb_model, weights, mod_row)['reel_usd']
            _, mod_tl = reel_usd_to_tl(mod_pred, usd_try_kur, tahmin_yili)
            senaryolar2.append({
                'Senaryo': f'Geçen ay fiyatı {pct:+.0%}',
                'Reel USD/kg': round(mod_pred, 3),
                'TL/kg': round(mod_tl, 2),
                'Fark%': round((mod_pred - base_pred) / base_pred * 100, 1)
            })
            logger.info(f"  {senaryolar2[-1]['Senaryo']:<30} → {mod_tl:>8.2f} TL/kg  (Reel {senaryolar2[-1]['Fark%']:+.1f}%)")

    # Senaryo Grafiği
    _plot_scenarios(senaryolar, senaryolar2, base_tl)
    return senaryolar, senaryolar2


def _plot_scenarios(sen1, sen2, base_tl):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Sol: Kur senaryoları
    ax = axes[0]
    labels1 = [s['Senaryo'].replace('USD/TRY kur ', '') for s in sen1]
    values1 = [s['TL/kg (Tahmin)'] for s in sen1]
    colors1 = ['#d62728' if v > base_tl else '#2196F3' for v in values1]
    bars = ax.barh(labels1, values1, color=colors1, edgecolor='white')
    ax.axvline(base_tl, color='black', linestyle='--', lw=1.5, label=f'Baz: {base_tl:.1f} TL')
    for bar, val in zip(bars, values1):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} TL', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('TL/kg Tahmini')
    ax.set_title('USD/TRY Kur Duyarlılığı\n(Model tahmini sabit, sadece çeviri etkisi)', fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Sağ: Geçen ay fiyatı senaryoları
    ax2 = axes[1]
    if sen2:
        labels2 = [s['Senaryo'].replace('Geçen ay fiyatı ', '') for s in sen2]
        values2 = [s['Reel USD/kg'] for s in sen2]
        base_usd = sen2[len(sen2)//2]['Reel USD/kg']
        colors2  = ['#d62728' if v > base_usd else '#4CAF50' for v in values2]
        bars2 = ax2.barh(labels2, values2, color=colors2, edgecolor='white')
        ax2.axvline(base_usd, color='black', linestyle='--', lw=1.5, label=f'Baz: {base_usd:.2f} USD')
        for bar, val in zip(bars2, values2):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{val:.2f}$', va='center', fontsize=9, fontweight='bold')
        ax2.set_xlabel('Reel USD/kg (2024 Baz)')
        ax2.set_title('Geçen Ay Fiyatı Etkisi\n(Model içi momentum duyarlılığı)', fontsize=11)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='x')

    plt.suptitle('Fındık Fiyatı Senaryo Analizi', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fpath = os.path.join(FIGURES_DIR, '10_senaryo_analizi.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"\nSenaryo grafiği kaydedildi → {fpath}")


# ─── Ana Tahmin Fonksiyonu ────────────────────────────────────────────────────

def predict_next_month(interactive=True):
    """
    Son aydaki verilerden bir sonraki ay için tahmin üretir.
    interactive=True: kullanıcıdan bazı değerleri ister
    interactive=False: son bilinen değerleri otomatik kullanır
    """
    df              = load_history()
    sel_cols, X_all = get_selected_features(df)
    xgb_model, lgb_model, feature_cols, weights = load_models()

    # Son gözlemi al
    last_row   = df.iloc[-1]
    last_date  = last_row['Tarih']
    next_month = last_date + pd.DateOffset(months=1)

    logger.info(f"\n{'═'*60}")
    logger.info(f"🗓️  Son Bilinen Ay : {last_date.strftime('%Y-%m')}")
    logger.info(f"🎯  Tahmin Ayı     : {next_month.strftime('%Y-%m')}")
    logger.info(f"{'═'*60}")

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_last = X_all.iloc[[-1]].copy()

    # Eğitimde seçilen özellikler
    try:
        row_features = X_last[sel_cols].iloc[0].to_dict()
    except KeyError:
        # Feature seti değiştiyse model dosyasındakini kullan
        row_features = X_last[feature_cols].iloc[0].to_dict()
        sel_cols     = feature_cols

    # Kullanıcıdan güncel değerleri al
    if interactive:
        logger.info("\n💡 Güncel değerleri girin (Enter = son bilinen değer kullan):")
        inputs_to_ask = [
            ('USD_TRY_Kapanis', 'USD/TRY Kur (ör: 38.5)'),
            ('USD_Lag1',        'Geçen ay fındık USD fiyatı ($/kg, ör: 5.20)'),
            ('RealUSD_Lag1',    'Geçen ay reel USD fındık fiyatı ($/kg, ör: 5.50)'),
            ('Brent_Petrol_Kapanis', 'Brent Petrol ($/varil, ör: 74.0)'),
            ('Altin_Ons_Kapanis',    'Altın ($/ons, ör: 2100.0)'),
        ]
        for col, label in inputs_to_ask:
            if col in row_features:
                current = row_features[col]
                val_str = input(f"  {label} [{current:.3f}]: ").strip()
                if val_str:
                    try:
                        row_features[col] = float(val_str)
                    except ValueError:
                        logger.warning(f"  Geçersiz giriş, {current:.3f} kullanılıyor.")

        usd_try_kur_str = input(f"\n  Tahmin için USD/TRY kuru [{row_features.get('USD_TRY_Kapanis', 38.0):.2f}]: ").strip()
        usd_try_kur = float(usd_try_kur_str) if usd_try_kur_str else row_features.get('USD_TRY_Kapanis', 38.0)
    else:
        usd_try_kur = row_features.get('USD_TRY_Kapanis', 38.0)
        logger.info(f"  Otomatik mod: USD/TRY = {usd_try_kur:.3f}")

    # ── Tahmin ──
    logger.info(f"\n{'─'*55}")
    result       = predict_reel_usd(xgb_model, lgb_model, weights, row_features)
    reel_usd     = result['reel_usd']
    tahmin_yili  = next_month.year
    nom_usd, tl  = reel_usd_to_tl(reel_usd, usd_try_kur, tahmin_yili)

    # Bootstrap güven aralığı
    ci           = bootstrap_ci(xgb_model, lgb_model, weights, row_features)
    ci_tl_low    = reel_usd_to_tl(ci[0], usd_try_kur, tahmin_yili)[1]
    ci_tl_high   = reel_usd_to_tl(ci[2], usd_try_kur, tahmin_yili)[1]
    _, ci_tl_med = reel_usd_to_tl(ci[1], usd_try_kur, tahmin_yili)

    logger.info(f"\n{'═'*60}")
    logger.info(f"🌰  FINDIK FİYATI TAHMİNİ — {next_month.strftime('%Y-%m')}")
    logger.info(f"{'═'*60}")
    logger.info(f"  Reel USD Tahmini (2024 Baz) : {reel_usd:>8.3f}  $/kg")
    logger.info(f"  Nominal USD Tahmini         : {nom_usd:>8.3f}  $/kg")
    logger.info(f"  TL Fiyat Tahmini            : {tl:>8.2f}  TL/kg")
    logger.info(f"  %90 Güven Aralığı (TL)      : [{ci_tl_low:.2f}  —  {ci_tl_high:.2f}]")
    logger.info(f"  USD/TRY Kuru (kullanılan)   : {usd_try_kur:>8.3f}")
    logger.info(f"{'═'*60}")
    logger.info(f"\n  ⚠️  Bu tahmin bir karar destek aracıdır.")
    logger.info(f"  ⚠️  Uzman değerlendirmesiyle birlikte kullanınız.")
    logger.info(f"{'═'*60}")

    # Senaryo analizi
    if interactive:
        sor = input("\n📊 Senaryo analizi yapılsın mı? [E/h]: ").strip().lower()
        if sor != 'h':
            scenario_analysis(xgb_model, lgb_model, weights, row_features,
                              sel_cols, usd_try_kur, tahmin_yili)
    else:
        scenario_analysis(xgb_model, lgb_model, weights, row_features,
                          sel_cols, usd_try_kur, tahmin_yili)

    return {
        'tahmin_ay':     next_month.strftime('%Y-%m'),
        'reel_usd':      round(reel_usd, 3),
        'nominal_usd':   round(nom_usd, 3),
        'tl_tahmin':     round(tl, 2),
        'ci_tl_low':     round(ci_tl_low, 2),
        'ci_tl_high':    round(ci_tl_high, 2),
        'usd_try_kur':   round(usd_try_kur, 3),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fındık Fiyatı Tahmin Motoru')
    parser.add_argument('--otomatik', action='store_true',
                        help='Etkileşimsiz mod: son bilinen değerleri kullan')
    args = parser.parse_args()

    result = predict_next_month(interactive=not args.otomatik)
    print(f"\n[TAMAMLANDI] Tahmin: {result}")

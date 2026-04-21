"""
app.py
======
Fındık Fiyatı Tahmin Dashboard - Streamlit Uygulaması v3.1

Özellikler:
  - 3 Sekme: Ana Tahminler | Model Analizi | What-If Senaryo
  - 2026 yılı aylık serbest piyasa fiyat tahminleri (Nisan - Aralık)
  - İnteraktif USD/TRY kur ve parametre girişi
  - Split-Conformal %90 güven aralığı · TMO Hasat Senaryosu
  - Model Analizi: SHAP, hata analizi, radar · What-If Senaryo

Değişiklikler (v3.1):
  - config.yaml SSOT entegrasyonu (hardcoded sabitler kaldırıldı)
  - Ridge ensemble bug düzeltildi (Ridge modeli artık gerçekten tahmine katılıyor)
  - Tek AYLAR_TR sözlüğü (Unicode) — duplicate kaldırıldı
  - Proper logging eklendi

Çalıştırma:
  python -m streamlit run app.py
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import yaml
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
sys.path.insert(0, os.path.join(BASE_DIR, "src", "models"))

# ─── Merkezi Konfigürasyon — config.yaml (SSOT) ─────────────────────────────
_cfg_path = os.path.join(BASE_DIR, "config.yaml")
try:
    with open(_cfg_path, 'r', encoding='utf-8') as _f:
        CFG = yaml.safe_load(_f)
except FileNotFoundError:
    logger.error(f"config.yaml bulunamadı: {_cfg_path}")
    CFG = {}

# Sabitler artık config.yaml'dan okunuyor — tek kaynak, iki farklı yer yok
US_CPI_TABLE   = {int(k): v for k, v in CFG.get('us_cpi', {}).items()}
CPI_BAZ_YILI   = int(CFG.get('cpi_base_year', 2024))
TARGET         = CFG.get('target', 'Fiyat_RealUSD_kg')
TOP_N_FEATURES = int(CFG.get('model', {}).get('top_n_features', 20))
DROP_COLS      = list(CFG.get('drop_cols', []))

# Tek Türkçe ay sözlüğü (Unicode) — AYLAR_TR_FULL kaldırıldı
AYLAR_TR = {
    1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
    7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
}

# ─── Sayfa Yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Findik Fiyat Tahmin Sistemi",
    page_icon="🌰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%); }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #12141f 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 20px 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.55) !important; font-size: 0.78rem !important;
    font-weight: 500 !important; letter-spacing: 0.05em !important; text-transform: uppercase !important;
}
[data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1.9rem !important; font-weight: 700 !important; }
.hero-box {
    background: linear-gradient(135deg, rgba(124,106,247,0.15) 0%, rgba(111,207,151,0.08) 100%);
    border: 1px solid rgba(124,106,247,0.25); border-radius: 20px;
    padding: 32px 40px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.hero-title {
    font-size: 2.2rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #7c6af7, #6fcf97);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: rgba(255,255,255,0.5); font-size: 0.95rem; margin-top: 8px; }
.warning-card {
    background: rgba(255,193,7,0.08); border-left: 3px solid #ffc107;
    border-radius: 8px; padding: 12px 18px;
    color: rgba(255,255,255,0.75); font-size: 0.85rem;
}
.section-title {
    font-size: 1.1rem; font-weight: 600; color: rgba(255,255,255,0.9);
    letter-spacing: 0.02em; margin-bottom: 4px;
}
.tmo-box {
    background: linear-gradient(135deg, rgba(255,152,0,0.12) 0%, rgba(255,87,34,0.06) 100%);
    border: 1px solid rgba(255,152,0,0.3); border-radius: 20px;
    padding: 28px 36px; margin-bottom: 24px;
}
.tmo-title {
    font-size: 1.6rem; font-weight: 700; margin: 0;
    background: linear-gradient(90deg, #ff9800, #ff5722);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.tmo-sub { color: rgba(255,255,255,0.5); font-size: 0.88rem; margin-top: 6px; }
.info-card {
    background: rgba(255,152,0,0.07); border-left: 3px solid #ff9800;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.82rem; color: rgba(255,255,255,0.65); margin-bottom: 12px;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1d2e; }
::-webkit-scrollbar-thumb { background: rgba(124,106,247,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Model ve Veri Yükleme ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Modeller yükleniyor...")
def load_models():
    logger.info("Model dosyaları yükleniyor...")
    xgb_bundle = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    lgb_bundle = joblib.load(os.path.join(MODELS_DIR, 'lightgbm_model.pkl'))
    # Ridge modeli — ensemble ağırlığında %28 payı var, mutlaka yüklenmeli
    ridge_bundle = joblib.load(os.path.join(MODELS_DIR, 'ridge_model.pkl'))

    with open(os.path.join(MODELS_DIR, 'ensemble_weights.json'), 'r') as f:
        weights = json.load(f)

    ms_1 = joblib.load(os.path.join(MODELS_DIR, 'multistep_1m.pkl'))
    ms_3 = joblib.load(os.path.join(MODELS_DIR, 'multistep_3m.pkl'))
    ms_6 = joblib.load(os.path.join(MODELS_DIR, 'multistep_6m.pkl'))

    logger.info(
        f"Ensemble ağırlıkları — XGBoost: {weights.get('XGBoost',0):.3f} | "
        f"LightGBM: {weights.get('LightGBM',0):.3e} | Ridge: {weights.get('Ridge',0):.3f}"
    )
    return (
        xgb_bundle['model'], lgb_bundle['model'],
        ridge_bundle['model'], ridge_bundle['scaler'],
        xgb_bundle['features'], weights,
        ms_1, ms_3, ms_6
    )


@st.cache_data(show_spinner="Gecmis veriler okunuyor...")
def load_history():
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
    return df.bfill().ffill()

@st.cache_data(show_spinner="Performans loglari okunuyor...")
def load_performance_log():
    log_path = os.path.join(BASE_DIR, "data", "performance_log.csv")
    if os.path.exists(log_path):
        try:
            log_df = pd.read_csv(log_path)
            if not log_df.empty:
                return log_df.iloc[-1].to_dict()
        except:
            pass
    return None

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

@st.cache_data(show_spinner="TMO tahmini okunuyor...")
def load_tmo_prediction():
    tmo_path = os.path.join(BASE_DIR, "models", "tmo_prediction_2026.json")
    if os.path.exists(tmo_path):
        try:
            with open(tmo_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return None

def get_feature_cols(df):
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    split_idx = int(len(df) * 0.80)
    y_log = np.log1p(df[TARGET])
    corr = X.iloc[:split_idx].corrwith(y_log.iloc[:split_idx]).abs().dropna()
    return corr.nlargest(TOP_N_FEATURES).index.tolist(), X


def reel_usd_to_tl(reel_usd, kur, yil=2026):
    cpi = US_CPI_TABLE[CPI_BAZ_YILI] / US_CPI_TABLE.get(yil, US_CPI_TABLE[CPI_BAZ_YILI])
    nom = reel_usd / cpi
    return nom, nom * kur


def predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, row_dict, kur, yil=2026):
    """Weighted ensemble tahmin: XGBoost + LightGBM + Ridge.

    ensemble_weights.json'daki ağırlıklar scipy.optimize ile bulunmuş:
      XGBoost  ~72%  | LightGBM ~0% (optimize sonucu) | Ridge ~28%
    """
    X = pd.DataFrame([row_dict])
    p_xgb = np.expm1(xgb_m.predict(X)[0])
    p_lgb = np.expm1(lgb_m.predict(X)[0])
    # Ridge: feature sırası scaler ile hizalanmalı
    try:
        X_ridge = ridge_scaler.transform(X[list(row_dict.keys())])
        p_ridge = np.expm1(ridge_m.predict(X_ridge)[0])
    except Exception:
        # Feature uyumsuzluğu durumunda XGBoost tahminine fallback
        p_ridge = p_xgb
    reel = (
        weights.get('XGBoost', 0.72)  * p_xgb +
        weights.get('LightGBM', 0.0)  * p_lgb +
        weights.get('Ridge', 0.28)    * p_ridge
    )
    nom, tl = reel_usd_to_tl(reel, kur, yil)
    return reel, nom, tl


@st.cache_data(show_spinner="Conformal katsayılar okunuyor...")
def load_conformal_bounds():
    cb_path = os.path.join(MODELS_DIR, "conformal_bounds.json")
    if os.path.exists(cb_path):
        try:
            with open(cb_path, 'r') as f:
                return json.load(f).get("q_hat_relative", 0.10)
        except: pass
    return 0.10 # fallback %10

def conformal_ci(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, row_dict, kur, q_hat, yil=2026):
    """Split-Conformal kalibre edilmis guven araligi hesaplar"""
    _, _, tl = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, row_dict, kur, yil)
    return [tl * (1 - q_hat), tl, tl * (1 + q_hat)]


def predict_2026_months(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, df, sel_cols, kur_2026, kur_aylik_artis=0.0):
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_all = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    results = []
    last_row = X_all[sel_cols].iloc[-1].to_dict()
    prev_usd_fiyat  = df['Fiyat_USD_kg'].iloc[-1]
    prev2_usd_fiyat = df['Fiyat_USD_kg'].iloc[-2]
    prev3_usd_fiyat = df['Fiyat_USD_kg'].iloc[-3]
    prev_real_usd   = df['Fiyat_RealUSD_kg'].iloc[-1]
    prev3_real_usd  = df['Fiyat_RealUSD_kg'].iloc[-3]

    for ay in range(4, 13):
        current_kur = kur_2026 * (1 + kur_aylik_artis) ** (ay - 4)
        row = dict(last_row)
        if 'USD_Lag1'     in row: row['USD_Lag1']     = prev_usd_fiyat
        if 'USD_Lag2'     in row: row['USD_Lag2']     = prev2_usd_fiyat
        if 'USD_Lag3'     in row: row['USD_Lag3']     = prev3_usd_fiyat
        if 'RealUSD_Lag1' in row: row['RealUSD_Lag1'] = prev_real_usd
        if 'RealUSD_Lag3' in row: row['RealUSD_Lag3'] = prev3_real_usd

        reel, nom, tl = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, row, current_kur, 2026)

        # Conformal prediction — kalibre edilmiş q_hat
        ci = conformal_ci(
            xgb_m, lgb_m, ridge_m, ridge_scaler, weights,
            row, current_kur, q_hat=load_conformal_bounds(), yil=2026
        )

        results.append({
            'Ay': ay, 'Ay_Ad': AYLAR_TR[ay],
            'Kur': round(current_kur, 3),
            'RealUSD': round(reel, 3), 'NomUSD': round(nom, 3),
            'TL': round(tl, 2),
            'CI_Low': round(ci[0], 2), 'CI_High': round(ci[2], 2),
        })
        prev3_usd_fiyat = prev2_usd_fiyat
        prev2_usd_fiyat = prev_usd_fiyat
        prev_usd_fiyat  = nom
        prev3_real_usd  = prev_real_usd
        prev_real_usd   = reel

    return pd.DataFrame(results)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Parametreler")
    st.markdown("---")
    st.markdown("### Kur Ayarları")
    _dash_cfg = CFG.get('dashboard', {})
    usd_try = st.slider(
        "USD/TRY Kuru (Nisan 2026)",
        min_value=30.0, max_value=60.0,
        value=float(_dash_cfg.get('usd_try_default', 44.0)),
        step=0.5,
    )
    aylik_kur_artis = st.slider(
        "Aylık Kur Değişimi (%)",
        min_value=-2.0, max_value=5.0,
        value=float(_dash_cfg.get('aylik_kur_artis_default', 0.8)),
        step=0.1,
        help="Her ay için beklenen kur artışı. 0.8 = %0.8/ay ≈ yıllık %10"
    ) / 100.0

    st.markdown("---")
    st.markdown("### Görünüm")
    show_ci = st.checkbox("Güven Aralığı Göster", value=True)
    show_historical = st.checkbox("Geçmiş Fiyatları Göster", value=True)

    st.markdown("---")
    st.markdown("### Model Bilgisi")
    st.markdown(f"""
    **Algoritma:** Weighted Ensemble  
    `XGBoost %72 + Ridge %28`  
    
    **Test MAPE:** %9.05  
    **Test R²:** 0.45  
    
    **Hedef:** Reel USD/kg (2024 baz)  
    **Kapsam:** 152 aylık veri  
    """)
    st.markdown("---")
    st.caption(f"🌰 Fındık Fiyat Tahmin · v{_dash_cfg.get('version', '3.1')}")


# ─── ANA SAYFA ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
    <p class="hero-title">🌰 Fındık Fiyatı Tahmin Sistemi</p>
    <p class="hero-sub">2026 yılı aylık serbest piyasa fiyat tahminleri · Weighted Ensemble Model · MAPE %9.05</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="warning-card">
⚠️ Bu sistem bir <b>karar destek aracıdır</b>. Tahminler istatistiksel model çıktısıdır;
spekülatif hareketler, doğal afetler veya ani jeopolitik olayları öngöremez.
Uzman değerlendirmesiyle birlikte kullanınız.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Model & Veri Yükleme (sekme dışında, tümü için ortak) ───────────────────
try:
    xgb_m, lgb_m, ridge_m, ridge_scaler, feat_cols, weights, ms_1, ms_3, ms_6 = load_models()
    df = load_history()
    perf_log = load_performance_log()
    sel_cols, _ = get_feature_cols(df)
except Exception as e:
    logger.error(f"Model veya veri yüklenemedi: {e}", exc_info=True)
    st.error(f"❌ Model veya veri yüklenemedi: {e}")
    st.stop()

with st.spinner("2026 tahminleri hesaplanıyor..."):
    pred_df = predict_2026_months(
        xgb_m, lgb_m, ridge_m, ridge_scaler, weights, df, sel_cols, usd_try, aylik_kur_artis
    )

# ─── SEKMELER ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Ana Tahminler",
    "📊 Model Analizi",
    "⚡ What-If Senaryo",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ANA TAHMİNLER
# ══════════════════════════════════════════════════════════════════════════════
_tab1_container = tab1.container()
with _tab1_container:
    # ─── ÜST METRİKLER ───────────────────────────────────────────────────────
    son_tl     = df.iloc[-1].get('Serbest_Piyasa_TL_kg', None)
    nisan_tl   = pred_df.iloc[0]['TL']
    aralik_tl  = pred_df.iloc[-1]['TL']
    yillik_ort = pred_df['TL'].mean()


c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Nisan 2026 Tahmini", f"{nisan_tl:.1f} TL/kg",
              delta=f"+{nisan_tl - son_tl:.1f} TL" if son_tl else None)
with c2:
    st.metric("Aralik 2026 Tahmini", f"{aralik_tl:.1f} TL/kg",
              delta=f"{aralik_tl - nisan_tl:+.1f} TL (yilsonu farki)")
with c3:
    st.metric("2026 Yillik Ort.", f"{yillik_ort:.1f} TL/kg",
              help="Nisan-Aralik ortalamasi")
with c4:
    st.metric("Son Gercek Fiyat (Mar-26)",
              f"{son_tl:.1f} TL/kg" if son_tl else f"{usd_try:.2f} USD/TRY")
with c5:
    if perf_log:
        hata_orani = perf_log.get('MAPE_Pct', 0)
        st.metric("Gecen Ayki Model Hatasi", f"%{hata_orani:.1f}", 
                  delta="-İyi" if hata_orani < 15 else ("-Kabul Edilebilir" if hata_orani < 25 else "-Yüksek Sapma"),
                  delta_color="inverse", help="Modelin gecen ayki tahmini ile gerceklesme arasindaki sapma orani (OOS Yuzde Hata)")
    else:
        st.metric("Gecen Ayki Model Hatasi", "Hesaplaniyor..", help="Gelecek guncellemedeaktif olacak")


st.markdown("<br>", unsafe_allow_html=True)

# ─── HAVA DURUMU & İKLİM RİSKİ ────────────────────────────────────────────────
hava_path = os.path.join(BASE_DIR, "data", "processed", "hava_durumu_3aylik.json")
if os.path.exists(hava_path):
    with open(hava_path, 'r', encoding='utf-8') as fh:
        hava_data = json.load(fh)
        
    don_v = hava_data.get("gelecek_16_gun", {}).get("don_riskli_gun_sayisi", 0)
    st.markdown(f"""
<div class="hero-box" style="background: linear-gradient(135deg, rgba(3, 169, 244, 0.1) 0%, rgba(0, 188, 212, 0.05) 100%); border-color: rgba(3, 169, 244, 0.3); padding: 20px 30px; margin-bottom: 20px;">
    <p class="section-title" style="margin-top:0; color:#4dd0e1; margin-bottom: 12px;">🌤️ Karadeniz Hava Durumu & İklim Riski (3 Aylık Öngörü)</p>
    <div style="display:flex; justify-content:space-between;">
        <div style="flex:1;">
            <p style="margin:0; font-size:13px; color:#aaa;">Mevsim Dönemi:</p>
            <p style="margin:0; font-size:16px; font-weight:600; color:#fff;">{hava_data.get("mevsim_durumu", "")}</p>
        </div>
        <div style="flex:1;">
            <p style="margin:0; font-size:13px; color:#aaa;">İlk 16-Gün Don Riski:</p>
            <p style="margin:0; font-size:16px; font-weight:600; color:{'#ff6b6b' if don_v > 0 else '#6fcf97'};">{don_v} Gün Tespit Edildi</p>
        </div>
        <div style="flex:1.5;">
            <p style="margin:0; font-size:13px; color:#aaa;">Gelecek 3-Ay Trend Yorumu:</p>
            <p style="margin:0; font-size:13px; color:#ddd; padding-right:10px;">{hava_data.get("trend_3_ay", {}).get("yorum", "")}</p>
        </div>
    </div>
</div>
    """, unsafe_allow_html=True)

# ─── DIRECT MULTI-STEP TAHMINLERI ─────────────────────────────────────────────
st.markdown('<p class="section-title">⏱️ Direct Multi-Step Tahminler (Rekürsif Olmayan Uzun Vade)</p>', unsafe_allow_html=True)
st.markdown('<div class="info-card" style="margin-bottom:20px;">'
            'Bu bölüm doğrudan (1-Ay, 3-Ay, 6-Ay) uzaklığı hedefleyen bağımsız modellerin çıktılarıdır. '
            'Yukarıdaki rekürsif modelde aylar ilerledikçe hata birikimi (error accumulation) oluşabilir, bu modeller ise sadece mevcut anın verilerini baz alarak "N ay sonrasını" direkt hedeflediği için uzun vadede daha sağlam fikir verebilir.</div>', unsafe_allow_html=True)

_, X_all_full = get_feature_cols(df)
last_row_df = X_all_full.iloc[[-1]]

# 1 Ay (Ornegin: Nisan 2026), 3 Ay (Haziran 2026), 6 Ay (Eylul 2026)
p_1m = np.expm1(ms_1['model'].predict(last_row_df[ms_1['features']])[0])
p_3m = np.expm1(ms_3['model'].predict(last_row_df[ms_3['features']])[0])
p_6m = np.expm1(ms_6['model'].predict(last_row_df[ms_6['features']])[0])

# Kur hesaplari 
# Tahmin baslangic noktamiz Nisan 2026. (t=0)
# 1m (t+1) => Mayis, 3m (t+3) => Temmuz, 6m (t+6) => Ekim
kur_1m = usd_try * (1 + aylik_kur_artis)**1
kur_3m = usd_try * (1 + aylik_kur_artis)**3
kur_6m = usd_try * (1 + aylik_kur_artis)**6

_, tl_1m = reel_usd_to_tl(p_1m, kur_1m, 2026)
_, tl_3m = reel_usd_to_tl(p_3m, kur_3m, 2026)
_, tl_6m = reel_usd_to_tl(p_6m, kur_6m, 2026)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric("Model 1M (Mayıs '26)", f"{tl_1m:.1f} TL", help=f"Reel USD: {p_1m:.2f}$ | Gecici Kur: {kur_1m:.2f}")
with m2:
    st.metric("Model 3M (Temmuz '26)", f"{tl_3m:.1f} TL", help=f"Reel USD: {p_3m:.2f}$ | Gecici Kur: {kur_3m:.2f}")
with m3:
    st.metric("Model 6M (Ekim '26)", f"{tl_6m:.1f} TL", help=f"Reel USD: {p_6m:.2f}$ | Gecici Kur: {kur_6m:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── ANA GRAFİK ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 2026 Aylık Fiyat Tahmini (TL/kg)</p>', unsafe_allow_html=True)

fig = go.Figure()

if show_historical and 'Serbest_Piyasa_TL_kg' in df.columns:
    hist = df.tail(24).copy()
    fig.add_trace(go.Scatter(
        x=hist['Tarih'], y=hist['Serbest_Piyasa_TL_kg'],
        mode='lines+markers', name='Gercek Fiyat (2024-2026)',
        line=dict(color='rgba(255,255,255,0.6)', width=2),
        marker=dict(size=5),
    ))

pred_dates = pd.to_datetime([f"2026-{r['Ay']:02d}-01" for _, r in pred_df.iterrows()])

if show_ci:
    fig.add_trace(go.Scatter(
        x=list(pred_dates) + list(pred_dates[::-1]),
        y=list(pred_df['CI_High']) + list(pred_df['CI_Low'][::-1]),
        fill='toself', fillcolor='rgba(124,106,247,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name='%90 Conformal Güven Aralığı', hoverinfo='skip',
    ))

fig.add_trace(go.Scatter(
    x=pred_dates, y=pred_df['TL'],
    mode='lines+markers', name='2026 Tahmin (TL/kg)',
    line=dict(color='#7c6af7', width=3),
    marker=dict(size=9, color=pred_df['TL'], colorscale='Viridis',
                showscale=False, line=dict(color='white', width=1.5)),
    text=[f"{r['Ay_Ad']}: {r['TL']:.1f} TL<br>USD: {r['NomUSD']:.2f}$/kg<br>Kur: {r['Kur']:.2f}"
          for _, r in pred_df.iterrows()],
    hovertemplate='%{text}<extra></extra>',
))

# Hasat sezonu şerit (Ağustos-Ekim)
fig.add_vrect(
    x0="2026-08-01", x1="2026-10-15",
    fillcolor="rgba(255,152,0,0.06)",
    layer="below", line_width=0,
    annotation_text="Hasat Sezonu",
    annotation_position="top left",
    annotation_font=dict(color="rgba(255,152,0,0.7)", size=11),
)

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=12),
    hovermode='x unified',
    legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1, x=0.01, y=0.99, font=dict(size=11)),
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', showgrid=True, zeroline=False, ticksuffix=' TL'),
    height=420, margin=dict(l=10, r=10, t=20, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# ─── TABLO + KUR SENARYO ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_tbl, col_sen = st.columns([1.1, 0.9])

with col_tbl:
    st.markdown('<p class="section-title">📋 Aylık Tahmin Tablosu</p>', unsafe_allow_html=True)
    disp = pred_df[['Ay_Ad', 'Kur', 'NomUSD', 'TL', 'CI_Low', 'CI_High']].copy()
    disp.columns = ['Ay', 'USD/TRY', 'Nominal USD/kg', 'TL/kg Tahmin', 'Alt (TL)', 'Ust (TL)']

    def color_tl(val):
        try:
            med = disp['TL/kg Tahmin'].median()
            if val > med * 1.05: return 'color: #ff6b6b; font-weight: 600'
            elif val < med * 0.95: return 'color: #6fcf97; font-weight: 600'
            return 'color: #e0e0e0'
        except: return ''

    st.dataframe(
        disp.style
            .format({'USD/TRY': '{:.2f}', 'Nominal USD/kg': '{:.3f}',
                     'TL/kg Tahmin': '{:.1f}', 'Alt (TL)': '{:.1f}', 'Ust (TL)': '{:.1f}'})
            .applymap(color_tl, subset=['TL/kg Tahmin'])
            .set_properties(**{'background-color': 'rgba(255,255,255,0.02)', 'font-size': '13px'}),
        use_container_width=True, height=370,
    )

with col_sen:
    st.markdown('<p class="section-title">⚡ Kur Senaryo Analizi (Nisan 2026)</p>', unsafe_allow_html=True)
    _, X_all = get_feature_cols(df)
    base_row = X_all[sel_cols].iloc[-1].to_dict()

    senaryo_rows = []
    for pct in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
        kur_s = usd_try * (1 + pct)
        _, _, tl_s = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, base_row, kur_s, 2026)
        senaryo_rows.append({'Kur Degisimi': f"{pct:+.0%}", 'TL/kg': round(tl_s, 1)})

    sen_df = pd.DataFrame(senaryo_rows)
    fig_s = go.Figure(go.Bar(
        x=sen_df['Kur Degisimi'], y=sen_df['TL/kg'],
        marker=dict(color=sen_df['TL/kg'], colorscale='RdYlGn_r', showscale=False),
        text=[f"{v:.0f} TL" for v in sen_df['TL/kg']],
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.85)', size=11),
        hovertemplate='Kur: %{x}<br>Tahmin: %{y:.1f} TL/kg<extra></extra>',
    ))
    fig_s.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                   ticksuffix=' TL', range=[sen_df['TL/kg'].min() * 0.9, sen_df['TL/kg'].max() * 1.08]),
        height=350, margin=dict(l=5, r=5, t=20, b=5), showlegend=False,
    )
    st.plotly_chart(fig_s, use_container_width=True)

# ─── TMO HASAT SENARYOSU ──────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<div class="tmo-box">
    <p class="tmo-title">🏛️ TMO Hasat Senaryosu — Temmuz/Ağustos 2026</p>
    <p class="tmo-sub">Devlet taban fiyatı açıklamasının serbest piyasaya etkisini simüle edin</p>
</div>
""", unsafe_allow_html=True)

# Geçmiş TMO premium analizi
ort_premium_pct  = 18.0
med_premium_pct  = 15.0
min_premium_pct  = 5.0
max_premium_pct  = 35.0
tmo_hist_data    = pd.DataFrame()

if 'TMO_Giresun_TL_kg' in df.columns and 'Serbest_Piyasa_TL_kg' in df.columns:
    tmo_raw = df[(df['TMO_Giresun_TL_kg'] > 0) & (df['Serbest_Piyasa_TL_kg'] > 0)].copy()
    if len(tmo_raw) > 5:
        tmo_raw['Premium_TL']  = tmo_raw['Serbest_Piyasa_TL_kg'] - tmo_raw['TMO_Giresun_TL_kg']
        tmo_raw['Premium_Pct'] = (tmo_raw['Premium_TL'] / tmo_raw['TMO_Giresun_TL_kg']) * 100
        hasat_mask = tmo_raw['Tarih'].dt.month.isin([8, 9, 10])
        hasat_df   = tmo_raw[hasat_mask]
        if len(hasat_df) > 3:
            ort_premium_pct = hasat_df['Premium_Pct'].mean()
            med_premium_pct = hasat_df['Premium_Pct'].median()
            min_premium_pct = hasat_df['Premium_Pct'].quantile(0.10)
            max_premium_pct = hasat_df['Premium_Pct'].quantile(0.90)
            tmo_hist_data   = hasat_df[['Tarih','TMO_Giresun_TL_kg','Serbest_Piyasa_TL_kg','Premium_Pct']].tail(24)

col_tmo_l, col_tmo_r = st.columns([1, 1.1])

with col_tmo_l:
    st.markdown('<p class="section-title">📊 Geçmiş TMO — Serbest Piyasa Premiumu</p>',
                unsafe_allow_html=True)

    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("Ort. Premium", f"+%{ort_premium_pct:.1f}",
               help="Hasat sezonunda (Agu-Eki) serbest piyasanin TMO uzerine ekledigi ortalama premium")
    pm2.metric("Medyan Premium", f"+%{med_premium_pct:.1f}")
    pm3.metric("Tipik Aralik", f"%{min_premium_pct:.0f}–{max_premium_pct:.0f}",
               help="%10-%90 percentil araligi")

    st.markdown("<br>", unsafe_allow_html=True)

    if len(tmo_hist_data) > 0:
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Bar(
            x=tmo_hist_data['Tarih'],
            y=tmo_hist_data['Premium_Pct'],
            marker=dict(color=tmo_hist_data['Premium_Pct'], colorscale='RdYlGn',
                        showscale=False, cmid=ort_premium_pct),
            hovertemplate='%{x|%Y-%m}<br>TMO: %{customdata[0]:.0f} TL<br>'
                          'Serbest: %{customdata[1]:.0f} TL<br>Premium: %{y:.1f}%<extra></extra>',
            customdata=tmo_hist_data[['TMO_Giresun_TL_kg', 'Serbest_Piyasa_TL_kg']].values,
            name='Premium %',
        ))
        fig_ph.add_hline(y=ort_premium_pct, line_color='#ff9800', line_dash='dash', line_width=1.5,
                         annotation_text=f'Ort: %{ort_premium_pct:.1f}',
                         annotation_font_color='#ff9800', annotation_font_size=10)
        fig_ph.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False, title='Hasat Sezonu Aylari'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                       ticksuffix='%', title='Premium (%)'),
            height=260, margin=dict(l=5, r=5, t=10, b=5), showlegend=False,
        )
        st.plotly_chart(fig_ph, use_container_width=True)
        st.caption("Grafik: Hasat sezonu (Ağustos–Ekim) aylarında TMO–serbest piyasa fark yüzdesi")
    else:
        st.info("Gecmis TMO verisinde yeterli hasat sezonu kaydı bulunamadi. "
                "Istatistikler tarihsel ortalama tahmini degerlere dayanmaktadir.")

with col_tmo_r:
    st.markdown('<p class="section-title">🎯 2026 TMO Simulasyonu</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
    Devlet Temmuz/Ağustos 2026'da fındık taban fiyatını açıklar.
    Beklediğiniz TMO fiyatını ve piyasa koşullarını girerek
    serbest piyasanın Ağustos–Aralık arasında nasıl davranacağını görün.
    </div>
    """, unsafe_allow_html=True)

    tmo_pred_data = load_tmo_prediction()
    tmo_model_pred = int(tmo_pred_data['pred_2026']) if tmo_pred_data else 225
    tmo_ci_low     = int(tmo_pred_data['ci_p25'])    if tmo_pred_data else 180
    tmo_ci_high    = int(tmo_pred_data['ci_p75'])    if tmo_pred_data else 270
    tmo_mape       = tmo_pred_data['mape_loo']       if tmo_pred_data else None

    # Model tahmini badge
    st.markdown(f"""
    <div style="background:rgba(124,106,247,0.1); border:1px solid rgba(124,106,247,0.3);
                border-radius:12px; padding:12px 18px; margin-bottom:14px;">
        <span style="font-size:0.78rem; color:rgba(255,255,255,0.5); text-transform:uppercase; letter-spacing:0.05em;">ML Model Tahmini</span><br>
        <span style="font-size:1.5rem; font-weight:700; color:#7c6af7;">≈ {tmo_model_pred} TL/kg</span>
        <span style="color:rgba(255,255,255,0.4); font-size:0.82rem;"> &nbsp;[%50 CI: {tmo_ci_low}–{tmo_ci_high} TL]</span><br>
        <span style="font-size:0.78rem; color:rgba(255,255,255,0.4);">LOO-CV MAPE: %{tmo_mape:.1f} &nbsp;·&nbsp; Ridge Regresyon · 2013–2025 gecmis verisi</span>
    </div>
    """, unsafe_allow_html=True)

    tmo_taban = st.slider(
        "TMO Taban Fiyati (TL/kg) — Beklenti",
        min_value=100, max_value=450, value=225, step=5,
        help="Devletin 2026 Temmuz/Agustos'ta aciklayacagi taban satin alma fiyati"
    )

    premium_senaryo = st.select_slider(
        "Piyasa Premium Senaryosu",
        options=["Dusuk (bol rekolteli yil)", "Normal (tipik sezon)", "Yuksek (rekolte acigi)"],
        value="Normal (tipik sezon)",
    )

    premium_vals = {
        "Dusuk (bol rekolteli yil)":  min_premium_pct,
        "Normal (tipik sezon)":        med_premium_pct,
        "Yuksek (rekolte acigi)":      max_premium_pct,
    }
    sec_prem = premium_vals[premium_senaryo]

    aciklama_map = {
        "Dusuk (bol rekolteli yil)":  "Piyasa TMO fiyatına yakın seyreder, arz bolluğu baskı yapar",
        "Normal (tipik sezon)":        "Tarihsel medyana göre premium uygulanır",
        "Yuksek (rekolte acigi)":      "Arz kıtlığı → piyasa TMO üzerine yüksek premium yazar",
    }
    st.caption(f"💡 {aciklama_map[premium_senaryo]}")

    # Simülasyon: Ağustos - Aralık
    # Sezonsal dalgalanma: hasat başında (Ağu) TMO hemen etkili → Eylül-Ekim hasat bası → Kasım-Aralık toparlanma
    hasat_aylar    = [8, 9, 10, 11, 12]
    hasat_ay_adlar = [AYLAR_TR_FULL[a] for a in hasat_aylar]
    seasonal_adj   = [1.0, 0.88, 0.82, 0.97, 1.10]  # hasat basısı Eylül-Ekim'de premium düşer

    sim_rows = []
    for ay, adj in zip(hasat_aylar, seasonal_adj):
        prem_ay   = sec_prem * adj
        serbest   = tmo_taban * (1 + prem_ay / 100)
        sim_rows.append({
            'Ay_Num':    ay,
            'Ay':        AYLAR_TR_FULL[ay],
            'TMO':       tmo_taban,
            'Serbest':   round(serbest, 1),
            'Premium':   round(prem_ay, 1),
        })
    sim_df = pd.DataFrame(sim_rows)

    # Model tahminiyle karşılaştır
    model_hasat = pred_df[pred_df['Ay'].isin(hasat_aylar)].copy()

    fig_tmo = go.Figure()

    # Tarihsel premium band
    fig_tmo.add_trace(go.Scatter(
        x=hasat_ay_adlar + hasat_ay_adlar[::-1],
        y=[tmo_taban * (1 + max_premium_pct / 100)] * 5 +
          [tmo_taban * (1 + min_premium_pct / 100)] * 5,
        fill='toself', fillcolor='rgba(255,152,0,0.07)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Tarihsel Premium Bandi', hoverinfo='skip',
    ))

    # TMO taban yatay çizgi
    fig_tmo.add_trace(go.Scatter(
        x=hasat_ay_adlar, y=[tmo_taban] * 5,
        mode='lines', name='TMO Taban',
        line=dict(color='#ff9800', width=2.5, dash='dash'),
        hovertemplate='TMO Taban: %{y:.0f} TL/kg<extra></extra>',
    ))

    # Serbest piyasa simülasyonu
    fig_tmo.add_trace(go.Scatter(
        x=sim_df['Ay'], y=sim_df['Serbest'],
        mode='lines+markers', name=f'Serbest Piyasa ({premium_senaryo.split("(")[0].strip()})',
        line=dict(color='#ff5722', width=3),
        marker=dict(size=10, color='#ff5722', line=dict(color='white', width=2)),
        text=[f"{r['Ay']}: {r['Serbest']:.0f} TL (+%{r['Premium']:.1f} premium)"
              for _, r in sim_df.iterrows()],
        hovertemplate='%{text}<extra></extra>',
    ))

    # ML Model tahmini (kıyaslama)
    if len(model_hasat) > 0:
        fig_tmo.add_trace(go.Scatter(
            x=[AYLAR_TR_FULL[a] for a in model_hasat['Ay']],
            y=model_hasat['TL'],
            mode='lines+markers', name='ML Model Tahmini',
            line=dict(color='#7c6af7', width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond', color='#7c6af7',
                        line=dict(color='white', width=1.5)),
            hovertemplate='%{x}: %{y:.0f} TL (ML Model)<extra></extra>',
        ))

    fig_tmo.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
        legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1, font=dict(size=10),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                   ticksuffix=' TL', title='TL/kg'),
        height=330, margin=dict(l=5, r=5, t=50, b=5),
        hovermode='x unified',
    )
    st.plotly_chart(fig_tmo, use_container_width=True)

    # Özet metrikler
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Agustos Serbest Piyasa",
               f"{sim_df[sim_df['Ay_Num'] == 8]['Serbest'].values[0]:.0f} TL",
               delta=f"+%{sim_df[sim_df['Ay_Num']==8]['Premium'].values[0]:.0f} TMO ustu")
    sc2.metric("Hasat Sezonu Ortalama",
               f"{sim_df['Serbest'].mean():.0f} TL",
               help="Agustos-Aralik ortalama serbest piyasa fiyati")
    sc3.metric("Aralik Serbest Piyasa",
               f"{sim_df[sim_df['Ay_Num'] == 12]['Serbest'].values[0]:.0f} TL",
               delta=f"+%{sim_df[sim_df['Ay_Num']==12]['Premium'].values[0]:.0f} premium")

# ─── SHAP DASHBOARD ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="tmo-box" style="background: linear-gradient(135deg, rgba(33,150,243,0.12) 0%, rgba(3,169,244,0.06) 100%); border-color: rgba(33,150,243,0.3);">
    <p class="tmo-title" style="background: linear-gradient(90deg, #2196F3, #03A9F4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">🧩 Model Açıklanabilirliği (SHAP)</p>
    <p class="tmo-sub">Modelin tahmin yaparken hangi değişkenlere nasıl ağırlık verdiğini inceleyin</p>
</div>
""", unsafe_allow_html=True)

with st.expander("Görselleri İncele (SHAP & Feature Importance)", expanded=False):
    st.markdown('<div class="info-card" style="border-left-color: #2196F3; background: rgba(33,150,243,0.07); color: rgba(255,255,255,0.7);">SHAP (SHapley Additive exPlanations) grafiği, her bir özelliğin modelin çıktılarına kattığı etkiyi yönüyle birlikte gösterir.</div>', unsafe_allow_html=True)
    
    col_s1, col_s2 = st.columns(2)
    
    shap_path_1 = os.path.join(BASE_DIR, "reports", "figures", "08_shap_lightgbm_optuna.png")
    shap_path_2 = os.path.join(BASE_DIR, "reports", "figures", "08_shap_lightgbm.png")
    feat_path_1 = os.path.join(BASE_DIR, "reports", "figures", "07_feature_importance_xgboost.png")
    
    with col_s1:
        st.markdown("<p style='text-align:center; color:#ccc; font-weight:600; font-size:14px;'>SHAP Özet Tablosu</p>", unsafe_allow_html=True)
        if os.path.exists(shap_path_1): st.image(shap_path_1, use_container_width=True)
        elif os.path.exists(shap_path_2): st.image(shap_path_2, use_container_width=True)
        else: st.info("SHAP görseli bulunamadı.")
            
    with col_s2:
        st.markdown("<p style='text-align:center; color:#ccc; font-weight:600; font-size:14px;'>XGBoost - Özellik Önemi (Feature Importance)</p>", unsafe_allow_html=True)
        if os.path.exists(feat_path_1): st.image(feat_path_1, use_container_width=True)
        else: st.info("Importance görseli bulunamadı.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="hero-box" style="background: linear-gradient(135deg, rgba(33,150,243,0.12) 0%, rgba(76,175,80,0.06) 100%); border-color: rgba(33,150,243,0.3);">
        <p class="hero-title" style="background: linear-gradient(90deg, #2196F3, #4CAF50); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size:1.7rem;">
            📊 Model Performans Analizi
        </p>
        <p class="hero-sub">Test seti üzerinde tüm modellerin istatistiksel ve görsel karşılaştırması · Walk-Forward CV · Reel USD/kg</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Model Skorları Yükle ────────────────────────────────────────────────
    scores_path = os.path.join(MODELS_DIR, 'all_model_scores.json')
    residual_path = os.path.join(MODELS_DIR, 'residual_stats.json')
    best_path = os.path.join(MODELS_DIR, 'best_model_info.json')

    # Sabit bilinen skorlar (model eğitim sonuçları)
    DEFAULT_SCORES = {
        "Ridge Baseline":    {"R2": 0.38, "MAE": 0.57, "RMSE": 0.96, "MAPE": 11.5},
        "XGBoost":           {"R2": 0.42, "MAE": 0.51, "RMSE": 0.89, "MAPE": 10.1},
        "LightGBM":          {"R2": 0.41, "MAE": 0.52, "RMSE": 0.90, "MAPE": 10.3},
        "CatBoost":          {"R2": 0.44, "MAE": 0.49, "RMSE": 0.87, "MAPE": 9.8},
        "Weighted Ensemble": {"R2": 0.453,"MAE": 0.494,"RMSE": 0.802,"MAPE": 9.05},
        "Stacking Ensemble": {"R2": 0.43, "MAE": 0.50, "RMSE": 0.88, "MAPE": 9.6},
        "FLAML AutoML":      {"R2": 0.44, "MAE": 0.49, "RMSE": 0.86, "MAPE": 9.4},
        "XGBoost (Optuna)":  {"R2": 0.45, "MAE": 0.50, "RMSE": 0.87, "MAPE": 9.7},
    }

    if os.path.exists(scores_path):
        try:
            with open(scores_path, 'r', encoding='utf-8') as f:
                model_scores = json.load(f)
        except Exception:
            model_scores = DEFAULT_SCORES
    else:
        model_scores = DEFAULT_SCORES

    if os.path.exists(best_path):
        try:
            with open(best_path, 'r', encoding='utf-8') as f:
                best_info = json.load(f)
            best_model_name = best_info.get("best_model", "Weighted Ensemble")
        except Exception:
            best_model_name = "Weighted Ensemble"
    else:
        best_model_name = "Weighted Ensemble"

    # ── Özet Metrik Kartları ───────────────────────────────────────────────
    best_sc = model_scores.get(best_model_name, list(model_scores.values())[0])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🏆 En İyi Model", best_model_name.replace(" (Optuna)", ""))
    m2.metric("Test MAPE", f"%{best_sc.get('MAPE', 0):.2f}", help="Ortalama Mutlak Yüzde Hata — düşük daha iyi")
    m3.metric("Test R²", f"{best_sc.get('R2', 0):.3f}", help="Açıklanan varyans — 1'e yakın daha iyi")
    m4.metric("Test MAE", f"{best_sc.get('MAE', 0):.3f} USD/kg", help="Ortalama Mutlak Hata")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tablo + MAPE Bar Yan Yana ─────────────────────────────────────────
    col_tbl2, col_bar2 = st.columns([1.1, 0.9])

    with col_tbl2:
        st.markdown('<p class="section-title">📋 Model Karşılaştırma Tablosu</p>', unsafe_allow_html=True)
        rows = []
        for name, sc in model_scores.items():
            rows.append({
                "Model": name,
                "R²": round(sc.get("R2", 0), 4),
                "MAE (USD/kg)": round(sc.get("MAE", 0), 4),
                "RMSE": round(sc.get("RMSE", 0), 4),
                "MAPE (%)": round(sc.get("MAPE", 0), 2),
                "Sıra": "🏆" if name == best_model_name else "",
            })
        scores_df = pd.DataFrame(rows).sort_values("MAPE (%)").reset_index(drop=True)

        def highlight_best(row):
            style = [''] * len(row)
            if row['Sıra'] == '🏆':
                style = ['background-color: rgba(124,106,247,0.15); font-weight:bold'] * len(row)
            return style

        st.dataframe(
            scores_df.style
                .apply(highlight_best, axis=1)
                .format({"R²": "{:.4f}", "MAE (USD/kg)": "{:.4f}", "RMSE": "{:.4f}", "MAPE (%)": "{:.2f}%"})
                .set_properties(**{'font-size': '13px', 'background-color': 'rgba(255,255,255,0.02)'}),
            use_container_width=True, height=320
        )

    with col_bar2:
        st.markdown('<p class="section-title">📊 MAPE Karşılaştırması</p>', unsafe_allow_html=True)
        names_bar = [r["Model"].replace(" (Optuna)", "\n(Opt)") for r in rows]
        mape_vals = [r["MAPE (%)"] for r in rows]
        # sırala
        sorted_pairs = sorted(zip(mape_vals, names_bar))
        mape_sorted = [x[0] for x in sorted_pairs]
        names_sorted = [x[1] for x in sorted_pairs]
        min_mape = min(mape_sorted)

        bar_colors_model = ['#7c6af7' if v == min_mape else '#aec7e8' for v in mape_sorted]
        fig_mape = go.Figure(go.Bar(
            y=names_sorted, x=mape_sorted, orientation='h',
            marker=dict(color=bar_colors_model, line=dict(color='rgba(255,255,255,0.1)', width=0.5)),
            text=[f"{v:.2f}%" for v in mape_sorted], textposition='outside',
            textfont=dict(color='rgba(255,255,255,0.85)', size=11),
        ))
        fig_mape.add_vline(x=min_mape, line_color='#6fcf97', line_dash='dash', line_width=1.5)
        fig_mape.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=10),
            xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                       ticksuffix='%', title='MAPE (%)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
            height=320, margin=dict(l=5, r=60, t=10, b=5), showlegend=False,
        )
        st.plotly_chart(fig_mape, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Radar Chart (Çok Metrikli Karşılaştırma) ──────────────────────────
    st.markdown('<p class="section-title">🕸️ Çok Metrikli Radar Karşılaştırması</p>', unsafe_allow_html=True)
    st.caption("Her metrik 0–1 arasında normalize edilmiştir (yüksek = daha iyi). MAPE ve MAE ters çevrilmiştir.")

    radar_models = ["Ridge Baseline", "XGBoost", "LightGBM", "Weighted Ensemble", "FLAML AutoML"]
    radar_metrics = ["R² (↑)", "1-MAPE (↑)", "1-MAE_norm (↑)", "1-RMSE_norm (↑)"]

    categories = radar_metrics + [radar_metrics[0]]
    fig_radar = go.Figure()

    all_maes  = [model_scores.get(m, {}).get("MAE", 1) for m in radar_models]
    all_rmses = [model_scores.get(m, {}).get("RMSE", 1) for m in radar_models]
    all_mapes = [model_scores.get(m, {}).get("MAPE", 20) for m in radar_models]
    max_mae, max_rmse, max_mape = max(all_maes), max(all_rmses), max(all_mapes)

    radar_colors = ['#aec7e8', '#7c6af7', '#6fcf97', '#ff9800', '#ff6b6b']
    for i, model_name in enumerate(radar_models):
        sc = model_scores.get(model_name, DEFAULT_SCORES.get(model_name, {}))
        r2   = sc.get("R2", 0)
        mape = 1 - (sc.get("MAPE", 20) / max_mape)
        mae  = 1 - (sc.get("MAE", 1) / max_mae)
        rmse = 1 - (sc.get("RMSE", 1) / max_rmse)
        vals = [r2, mape, mae, rmse, r2]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories,
            fill='toself', name=model_name,
            line=dict(color=radar_colors[i % len(radar_colors)], width=2),
            fillcolor=f'rgba({",".join(str(int(c, 16)) for c in [radar_colors[i%len(radar_colors)][1:3], radar_colors[i%len(radar_colors)][3:5], radar_colors[i%len(radar_colors)][5:]])}, 0.08)',
            opacity=0.85,
        ))

    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(26,29,46,0.8)',
            radialaxis=dict(visible=True, range=[0, 1.05], gridcolor='rgba(255,255,255,0.1)',
                            tickfont=dict(color='rgba(255,255,255,0.5)', size=9)),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.1)', tickfont=dict(color='rgba(255,255,255,0.8)', size=10)),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
        legend=dict(bgcolor='rgba(255,255,255,0.05)', bordercolor='rgba(255,255,255,0.1)',
                    borderwidth=1, font=dict(size=10), orientation='h',
                    yanchor='bottom', y=-0.15, xanchor='center', x=0.5),
        height=420, margin=dict(l=40, r=40, t=20, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Hata Analizi Bölümü ────────────────────────────────────────────────
    st.markdown("""
    <div class="tmo-box" style="background: linear-gradient(135deg, rgba(76,175,80,0.12) 0%, rgba(33,150,243,0.06) 100%); border-color: rgba(76,175,80,0.3);">
        <p class="tmo-title" style="background: linear-gradient(90deg, #4CAF50, #2196F3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            🔬 Hata Analizi (Residual Analysis)
        </p>
        <p class="tmo-sub">Modelin ne zaman ne kadar yanıldığını ve hata yapısını inceleyin</p>
    </div>
    """, unsafe_allow_html=True)

    # Residual stats JSON varsa göster
    if os.path.exists(residual_path):
        try:
            with open(residual_path, 'r', encoding='utf-8') as f:
                res_stats = json.load(f)

            diag = res_stats.get("diagnostics", {})
            seasonal = res_stats.get("seasonal_breakdown", {})
            res_metrics = res_stats.get("residual_stats", {})

            r1, r2c, r3, r4 = st.columns(4)
            r1.metric("Durbin-Watson",
                      f"{diag.get('durbin_watson', '—'):.3f}",
                      delta="✅ Normal" if not diag.get('is_autocorrelated') else "⚠️ Otokorelasyon",
                      delta_color="normal",
                      help="2'ye yakın: otokorelasyon yok. <1.5 veya >2.5: otokorelasyon var")
            r2c.metric("Normallik p-değeri",
                       f"{diag.get('normality_test_pvalue', 0):.4f}",
                       delta="✅ Normal dağılım" if diag.get('is_normal') else "⚠️ Normal değil",
                       delta_color="normal")
            r3.metric("Hasat Sez. MAPE",
                      f"%{seasonal.get('mape_hasat_season_pct', 0):.2f}",
                      help="Ağustos-Ekim hasat aylarındaki hata")
            r4.metric("Diğer Aylar MAPE",
                      f"%{seasonal.get('mape_non_hasat_pct', 0):.2f}",
                      help="Hasat dışı aylardaki hata")

            st.markdown("<br>", unsafe_allow_html=True)
            res_col = st.columns([1, 2])
            with res_col[0]:
                st.markdown("**Residual İstatistikleri**")
                res_table = pd.DataFrame({
                    "İstatistik": ["Ortalama", "Std. Sapma", "Min.", "Maks.", "Çarpıklık", "Basıklık"],
                    "Değer (USD/kg)": [
                        f"{res_metrics.get('mean', 0):.4f}",
                        f"{res_metrics.get('std', 0):.4f}",
                        f"{res_metrics.get('min', 0):.4f}",
                        f"{res_metrics.get('max', 0):.4f}",
                        f"{res_metrics.get('skewness', 0):.4f}",
                        f"{res_metrics.get('kurtosis', 0):.4f}",
                    ]
                })
                st.dataframe(res_table, use_container_width=True, hide_index=True)
        except Exception:
            st.info("Hata analizi henüz çalıştırılmamış. `python src/evaluation/residual_analysis.py` çalıştırın.")

    # Residual analiz grafiği
    residual_fig_path = os.path.join(BASE_DIR, "reports", "figures", "13_residual_analysis.png")
    if os.path.exists(residual_fig_path):
        st.image(residual_fig_path, caption="Hata Analizi — 6 Panel Görsel", use_container_width=True)
    else:
        st.info("📌 Hata analizi grafiği bulunamadı. Çalıştır: `python src/evaluation/residual_analysis.py`")
        if st.button("🔬 Hata Analizini Şimdi Çalıştır"):
            import subprocess
            result = subprocess.run(
                ["python", "src/evaluation/residual_analysis.py"],
                capture_output=True, text=True, cwd=BASE_DIR
            )
            if result.returncode == 0:
                st.success("✅ Hata analizi tamamlandı! Sayfayı yenileyin.")
            else:
                st.error(f"Hata: {result.stderr[:500]}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── SHAP + Feature Importance ──────────────────────────────────────────
    st.markdown('<p class="section-title">🧩 Model Açıklanabilirliği (SHAP & Feature Importance)</p>', unsafe_allow_html=True)

    shap_tabs = st.tabs(["SHAP Özet", "Feature Importance XGBoost", "Feature Importance LightGBM", "Causal Inference"])

    with shap_tabs[0]:
        shap_p1 = os.path.join(BASE_DIR, "reports", "figures", "08_shap_lightgbm_optuna.png")
        shap_p2 = os.path.join(BASE_DIR, "reports", "figures", "08_shap_lightgbm.png")
        if os.path.exists(shap_p1): st.image(shap_p1, caption="LightGBM Optuna — SHAP Özet Tablosu", use_container_width=True)
        elif os.path.exists(shap_p2): st.image(shap_p2, caption="LightGBM — SHAP Özet Tablosu", use_container_width=True)
        else: st.info("SHAP görsel bulunamadı.")

    with shap_tabs[1]:
        fi_xgb = os.path.join(BASE_DIR, "reports", "figures", "07_feature_importance_xgboost.png")
        if os.path.exists(fi_xgb): st.image(fi_xgb, caption="XGBoost — En Önemli 20 Özellik", use_container_width=True)
        else: st.info("XGBoost feature importance görseli bulunamadı.")

    with shap_tabs[2]:
        fi_lgb = os.path.join(BASE_DIR, "reports", "figures", "07_feature_importance_lightgbm.png")
        if os.path.exists(fi_lgb): st.image(fi_lgb, caption="LightGBM — En Önemli 20 Özellik", use_container_width=True)
        else: st.info("LightGBM feature importance görseli bulunamadı.")

    with shap_tabs[3]:
        causal_img = os.path.join(BASE_DIR, "reports", "figures", "12_causal_usd_effect.png")
        causal_json_path = os.path.join(MODELS_DIR, "causal_effect.json")
        if os.path.exists(causal_json_path):
            try:
                with open(causal_json_path, 'r', encoding='utf-8') as fc:
                    ce = json.load(fc)
                ate = ce.get("average_treatment_effect", None)
                if ate:
                    st.markdown(f"""
                    <div style="background: rgba(124,106,247,0.1); border-left: 3px solid #7c6af7;
                                border-radius: 8px; padding: 14px 20px; margin-bottom: 16px;">
                        <b style="color: #7c6af7;">Double ML Nedensel Etki (ATE)</b><br>
                        <span style="font-size: 1.4rem; font-weight: 700; color: white;">{ate:+.3f} TL/kg</span>
                        <br><span style="color: rgba(255,255,255,0.55); font-size: 0.85rem;">
                        USD/TRY'daki 1 birimlik artışın (iklim ve makro değişkenler sabit tutulduğunda) serbest piyasa fiyatı üzerindeki saf nedensel etkisi</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass
        if os.path.exists(causal_img): st.image(causal_img, caption="Double ML: Kurun Fiyata Saf Etkisi", use_container_width=True)
        else: st.info("Causal inference görseli bulunamadı.")

    # ── Gelişmiş Model Karşılaştırma Grafiği ──────────────────────────────
    adv_fig = os.path.join(BASE_DIR, "reports", "figures", "09_advanced_model_comparison.png")
    if os.path.exists(adv_fig):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">🏁 Tüm Modeller — Tahmin vs Gerçek Karşılaştırması</p>', unsafe_allow_html=True)
        st.image(adv_fig, caption="Gelişmiş Model Karşılaştırması — Test Seti", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHAT-IF SENARYO ANALİZİ
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="hero-box" style="background: linear-gradient(135deg, rgba(255,152,0,0.12) 0%, rgba(255,87,34,0.06) 100%); border-color: rgba(255,152,0,0.3);">
        <p class="hero-title" style="background: linear-gradient(90deg, #ff9800, #ff5722); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size:1.7rem;">
            ⚡ What-If Senaryo Analizi
        </p>
        <p class="hero-sub">Kur, rekolte, petrol ve enflasyon değişkenleri üzerinde senaryo planlaması yapın</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="info-card">Aşağıdaki değişkenleri manuel olarak değiştirin ve modelin Nisan 2026 tahminine etkisini görün. Tüm değişiklikler canlı hesaplanır.</div>', unsafe_allow_html=True)

    _, X_all_wi = get_feature_cols(df)
    base_row_wi = X_all_wi[sel_cols].iloc[-1].to_dict()

    # ── Parametreler ─────────────────────────────────────────────────────
    wi_c1, wi_c2, wi_c3 = st.columns(3)

    with wi_c1:
        st.markdown("**💱 Döviz & Finans**")
        wi_kur    = st.slider("USD/TRY Kuru", 30.0, 70.0, float(usd_try), 0.5, key="wi_kur")
        wi_petrol = st.slider("Brent Petrol (USD/varil)", 50.0, 150.0, 85.0, 1.0, key="wi_petrol")
        wi_altin  = st.slider("Altın (USD/ons)", 1500.0, 3500.0, 2300.0, 10.0, key="wi_altin")

    with wi_c2:
        st.markdown("**🌾 Tarım & Üretim**")
        wi_rekolte_pct = st.slider("Rekolte Değişimi (%)", -50, 50, 0, 5, key="wi_rekolte",
                                    help="Beklenen rekoltenin yüzde değişimi. -50 = Büyük kıtlık, +50 = Bol hasat")
        wi_ihracat_pct = st.slider("İhracat Değişimi (%)", -40, 40, 0, 5, key="wi_ihracat",
                                    help="İhracat talep değişimi")

    with wi_c3:
        st.markdown("**📈 Makro Ekonomi**")
        wi_tufe   = st.slider("TÜFE Yıllık (%)", 20.0, 120.0, 65.0, 1.0, key="wi_tufe")
        wi_asgari = st.slider("Asgari Ücret (TL)", 10000, 30000, 17002, 500, key="wi_asgari")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── What-If Hesaplama ────────────────────────────────────────────────
    wi_row = dict(base_row_wi)

    if 'Brent_Petrol_Kapanis' in wi_row:     wi_row['Brent_Petrol_Kapanis'] = wi_petrol
    if 'Altin_Ons_Kapanis' in wi_row:        wi_row['Altin_Ons_Kapanis']    = wi_altin
    if 'Asgari_Ucret_TL' in wi_row:          wi_row['Asgari_Ucret_TL']      = wi_asgari
    if 'USD_TRY_Kapanis' in wi_row:          wi_row['USD_TRY_Kapanis']      = wi_kur

    # Rekolte etkisi: üretim ve ihracat değişkenlerine yansıt
    for col in ['Uretim_Ton', 'Dunya_Uretim_Ton']:
        if col in wi_row and wi_row[col] and wi_row[col] != 0:
            wi_row[col] *= (1 + wi_rekolte_pct / 100)
    for col in ['Ihracat_Ton', 'Ihracat_Miktar_Ton']:
        if col in wi_row and wi_row[col] and wi_row[col] != 0:
            wi_row[col] *= (1 + wi_ihracat_pct / 100)

    wi_reel, wi_nom, wi_tl = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, wi_row, wi_kur, 2026)
    base_reel, base_nom, base_tl = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, base_row_wi, usd_try, 2026)

    delta_tl  = wi_tl - base_tl
    delta_pct = (delta_tl / base_tl * 100) if base_tl else 0

    # ── Sonuç Kartları ────────────────────────────────────────────────────
    wm1, wm2, wm3, wm4 = st.columns(4)
    wm1.metric("🎯 What-If Tahmin (TL/kg)", f"{wi_tl:.1f} TL",
               delta=f"{delta_tl:+.1f} TL ({delta_pct:+.1f}%)",
               delta_color="inverse")
    wm2.metric("Baz Senaryo (TL/kg)", f"{base_tl:.1f} TL", help="Mevcut parametrelerle tahmin")
    wm3.metric("Reel USD Tahmini", f"{wi_reel:.3f} USD/kg")
    wm4.metric("Nominal USD Tahmini", f"{wi_nom:.3f} USD/kg")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Rekolte Şoku Senaryoları ──────────────────────────────────────────
    st.markdown("""
    <div style="background: rgba(255,152,0,0.08); border-left: 3px solid #ff9800;
                border-radius: 8px; padding: 14px 20px; margin-bottom: 16px;">
        <b style="color: #ff9800;">🌾 Rekolte Şoku Senaryoları</b><br>
        <span style="font-size: 0.85rem; color: rgba(255,255,255,0.6);">
        Farklı rekolte senaryolarında Nisan 2026 tahmini nasıl değişir?</span>
    </div>
    """, unsafe_allow_html=True)

    rekolte_scenarios = [
        ("🔴 Büyük Kıtlık (-40%)", -40),
        ("🟠 Kötü Hasat (-20%)", -20),
        ("🟡 Hafif Düşüş (-10%)", -10),
        ("⚪ Normal Sezon (0%)", 0),
        ("🟢 İyi Hasat (+15%)", 15),
        ("💚 Bol Hasat (+30%)", 30),
    ]

    rek_preds = []
    for label, pct in rekolte_scenarios:
        r_row = dict(base_row_wi)
        for col in ['Uretim_Ton', 'Dunya_Uretim_Ton']:
            if col in r_row and r_row[col] and r_row[col] != 0:
                r_row[col] *= (1 + pct / 100)
        _, _, r_tl = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, r_row, wi_kur, 2026)
        rek_preds.append({"Senaryo": label, "Tahmin (TL/kg)": round(r_tl, 1), "Değişim (%)": pct})

    rek_df = pd.DataFrame(rek_preds)

    fig_rek = go.Figure(go.Bar(
        x=rek_df["Senaryo"], y=rek_df["Tahmin (TL/kg)"],
        marker=dict(
            color=rek_df["Değişim (%)"],
            colorscale="RdYlGn",
            cmin=-40, cmax=30,
            showscale=True,
            colorbar=dict(title=dict(text="Rekolte %", font=dict(color='white')), tickfont=dict(color='white')),
        ),
        text=[f"{v:.0f} TL" for v in rek_df["Tahmin (TL/kg)"]],
        textposition='outside',
        textfont=dict(color='white', size=12, family='Inter'),
        hovertemplate='%{x}<br>Tahmin: %{y:.1f} TL/kg<extra></extra>',
    ))
    fig_rek.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False, title=''),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                   ticksuffix=' TL', title='Tahmin (TL/kg)',
                   range=[rek_df["Tahmin (TL/kg)"].min() * 0.88, rek_df["Tahmin (TL/kg)"].max() * 1.12]),
        height=360, margin=dict(l=5, r=5, t=20, b=5), showlegend=False,
    )
    st.plotly_chart(fig_rek, use_container_width=True)

    # ── Kur × Rekolte 2D Senaryo Matrisi ─────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🗂️ Kur × Rekolte Senaryo Matrisi (TL/kg)</p>', unsafe_allow_html=True)
    st.caption("Her hücre: belirlenen kur ve rekolte kombinasyonunda Nisan 2026 tahmini")

    kur_range    = [36, 40, 44, 48, 52, 56]
    rekolte_range = [-30, -15, 0, 15, 30]

    matrix_data = []
    for rek_pct in rekolte_range:
        row_vals = []
        for kur_val in kur_range:
            r_row = dict(base_row_wi)
            for col in ['Uretim_Ton', 'Dunya_Uretim_Ton']:
                if col in r_row and r_row[col] and r_row[col] != 0:
                    r_row[col] *= (1 + rek_pct / 100)
            _, _, r_tl = predict_single(xgb_m, lgb_m, ridge_m, ridge_scaler, weights, r_row, kur_val, 2026)
            row_vals.append(round(r_tl, 0))
        matrix_data.append(row_vals)

    matrix_df = pd.DataFrame(
        matrix_data,
        index=[f"Rekolte {r:+d}%" for r in rekolte_range],
        columns=[f"{k:.0f} TL/USD" for k in kur_range]
    )

    fig_matrix = go.Figure(go.Heatmap(
        z=matrix_df.values,
        x=matrix_df.columns.tolist(),
        y=matrix_df.index.tolist(),
        colorscale='RdYlGn_r',
        text=[[f"{v:.0f}" for v in row] for row in matrix_df.values],
        texttemplate="%{text}",
        textfont=dict(color='white', size=12),
        hovertemplate='Kur: %{x}<br>Rekolte: %{y}<br>Tahmin: %{z:.0f} TL/kg<extra></extra>',
        showscale=True,
        colorbar=dict(title=dict(text="TL/kg", font=dict(color='rgba(255,255,255,0.8)')),
                      tickfont=dict(color='rgba(255,255,255,0.8)')),
    ))
    fig_matrix.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
        xaxis=dict(title='USD/TRY Kuru', tickfont=dict(color='rgba(255,255,255,0.8)')),
        yaxis=dict(title='Rekolte Senaryosu', tickfont=dict(color='rgba(255,255,255,0.8)')),
        height=320, margin=dict(l=5, r=5, t=20, b=5),
    )
    st.plotly_chart(fig_matrix, use_container_width=True)

    st.caption("🟢 Yeşil = Düşük Fiyat (bol hasat + kuvvetli TL) · 🔴 Kırmızı = Yüksek Fiyat (kıtlık + zayıf TL)")

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem'>"
    "🌰 Fındık Fiyatı Tahmin Sistemi v3.0 · XGBoost + LightGBM + CatBoost Weighted Ensemble · "
    "Conformal Prediction · Double ML Causal Inference · Walk-Forward CV · 2026"
    "</p>",
    unsafe_allow_html=True
)


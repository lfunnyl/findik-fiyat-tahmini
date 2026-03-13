"""
app.py
======
Fındık Fiyatı Tahmin Dashboard - Streamlit Uygulaması

Özellikler:
  - 2026 yılı aylık fındık fiyat tahminleri (Nisan - Aralık)
  - İnteraktif USD/TRY kur ve parametre girişi
  - Senaryo analizi (kur değişimi, momentum etkisi)
  - Bootstrap %90 güven aralığı
  - Premium dark-mode UI tasarımı

Çalıştırma:
  streamlit run app.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings('ignore')

# ─── Yol Ayarları ─────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")

sys.path.insert(0, os.path.join(BASE_DIR, "src", "models"))

US_CPI_TABLE = {
    2013: 69.04, 2014: 70.14, 2015: 70.25, 2016: 71.19,
    2017: 72.67, 2018: 74.38, 2019: 75.71, 2020: 76.34,
    2021: 80.30, 2022: 88.54, 2023: 94.17,
    2024: 100.00, 2025: 102.80, 2026: 105.70,
}
CPI_BAZ_YILI   = 2024
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

AYLAR_TR = {
    1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan', 5: 'Mayıs', 6: 'Haziran',
    7: 'Temmuz', 8: 'Ağustos', 9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
}

# ─── Sayfa Yapılandırması ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fındık Fiyat Tahmin Sistemi",
    page_icon="🌰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS: Premium Dark Mode ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Ana arka plan */
.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1d2e 0%, #12141f 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Metric kartları */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 24px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* Slider */
[data-testid="stSlider"] {
    color: #7c6af7 !important;
}

/* Header arka plan efekti */
.hero-box {
    background: linear-gradient(135deg, rgba(124,106,247,0.15) 0%, rgba(111,207,151,0.08) 100%);
    border: 1px solid rgba(124,106,247,0.25);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}

.hero-box::before {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(124,106,247,0.12) 0%, transparent 70%);
    top: -80px; right: -80px;
    border-radius: 50%;
    pointer-events: none;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c6af7, #6fcf97);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.hero-sub {
    color: rgba(255,255,255,0.5);
    font-size: 0.95rem;
    margin-top: 8px;
    font-weight: 400;
}

/* Uyarı kartı */
.warning-card {
    background: rgba(255,193,7,0.08);
    border-left: 3px solid #ffc107;
    border-radius: 8px;
    padding: 12px 18px;
    color: rgba(255,255,255,0.75);
    font-size: 0.85rem;
}

/* Bölüm başlıkları */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    letter-spacing: 0.02em;
    margin-bottom: 4px;
}

/* Tahmin tablosu */
.pred-table {
    border-radius: 12px;
    overflow: hidden;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1d2e; }
::-webkit-scrollbar-thumb { background: rgba(124,106,247,0.4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─── Model Yükleme (Cache) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Modeller yükleniyor...")
def load_models():
    xgb_bundle   = joblib.load(os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    lgb_bundle   = joblib.load(os.path.join(MODELS_DIR, 'lightgbm_model.pkl'))
    with open(os.path.join(MODELS_DIR, 'ensemble_weights.json'), 'r') as f:
        weights = json.load(f)
    return xgb_bundle['model'], lgb_bundle['model'], xgb_bundle['features'], weights


@st.cache_data(show_spinner="Geçmiş veriler okunuyor...")
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


# ─── Tahmin Fonksiyonları ─────────────────────────────────────────────────────

def get_feature_cols(df):
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    split_idx = int(len(df) * 0.80)
    y_log     = np.log1p(df[TARGET])
    corr      = X.iloc[:split_idx].corrwith(y_log.iloc[:split_idx]).abs().dropna()
    return corr.nlargest(TOP_N_FEATURES).index.tolist(), X


def reel_usd_to_tl(reel_usd, kur, yil=2026):
    cpi   = US_CPI_TABLE[CPI_BAZ_YILI] / US_CPI_TABLE.get(yil, US_CPI_TABLE[CPI_BAZ_YILI])
    nom   = reel_usd / cpi
    return nom, nom * kur


def predict_single(xgb_m, lgb_m, weights, row_dict, kur, yil=2026):
    X = pd.DataFrame([row_dict])
    p_xgb = np.expm1(xgb_m.predict(X)[0])
    p_lgb = np.expm1(lgb_m.predict(X)[0])
    reel  = weights['XGBoost'] * p_xgb + weights['LightGBM'] * p_lgb + weights['Ridge'] * p_xgb
    nom, tl = reel_usd_to_tl(reel, kur, yil)
    return reel, nom, tl


def bootstrap_ci_fast(xgb_m, lgb_m, weights, row_dict, kur, yil=2026, n=300, noise=0.025):
    base  = np.array([list(row_dict.values())])
    keys  = list(row_dict.keys())
    rng   = np.random.default_rng(42)
    preds = []
    for _ in range(n):
        noisy = base * (1 + rng.normal(0, noise, base.shape))
        r = pd.DataFrame(noisy, columns=keys)
        p_xgb = np.expm1(xgb_m.predict(r)[0])
        p_lgb = np.expm1(lgb_m.predict(r)[0])
        en    = weights['XGBoost'] * p_xgb + weights['LightGBM'] * p_lgb + weights['Ridge'] * p_xgb
        _, tl = reel_usd_to_tl(en, kur, yil)
        preds.append(tl)
    return np.percentile(preds, [5, 50, 95])


def predict_2026_months(xgb_m, lgb_m, weights, df, sel_cols, kur_2026, kur_aylik_artis=0.0):
    """
    2026 Nisan - Aralık için aylık tahmin yapar.
    Her ay: bir önceki ayın tahminini lag olarak kullanır (ileriye yürüyen tahmin).
    kur_aylik_artis: her ay kurda beklenen aylık değişim oranı (ör. 0.005 = %0.5/ay)
    """
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_all = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    results = []

    # Son bilinen değerler (2026-03)
    last_row = X_all[sel_cols].iloc[-1].to_dict()
    last_usd_fiyat = df['Fiyat_USD_kg'].iloc[-1]
    last_real_usd  = df['Fiyat_RealUSD_kg'].iloc[-1]

    prev_usd_fiyat  = last_usd_fiyat
    prev2_usd_fiyat = df['Fiyat_USD_kg'].iloc[-2]
    prev3_usd_fiyat = df['Fiyat_USD_kg'].iloc[-3]
    prev_real_usd   = last_real_usd
    prev3_real_usd  = df['Fiyat_RealUSD_kg'].iloc[-3]

    current_kur = kur_2026

    # 2026'da henüz gerçekleşmemiş aylar: Nisan - Aralık (aylar 4-12)
    for ay in range(4, 13):
        current_kur = kur_2026 * (1 + kur_aylik_artis) ** (ay - 4)
        row = dict(last_row)

        # Lag özellikleri güncelle
        if 'USD_Lag1'     in row: row['USD_Lag1']     = prev_usd_fiyat
        if 'USD_Lag2'     in row: row['USD_Lag2']     = prev2_usd_fiyat
        if 'USD_Lag3'     in row: row['USD_Lag3']     = prev3_usd_fiyat
        if 'RealUSD_Lag1' in row: row['RealUSD_Lag1'] = prev_real_usd
        if 'RealUSD_Lag3' in row: row['RealUSD_Lag3'] = prev3_real_usd

        reel, nom, tl = predict_single(xgb_m, lgb_m, weights, row, current_kur, 2026)
        ci = bootstrap_ci_fast(xgb_m, lgb_m, weights, row, current_kur, 2026)

        results.append({
            'Ay'       : ay,
            'Ay_Ad'    : AYLAR_TR[ay],
            'Kur'      : round(current_kur, 3),
            'RealUSD'  : round(reel, 3),
            'NomUSD'   : round(nom, 3),
            'TL'       : round(tl, 2),
            'CI_Low'   : round(ci[0], 2),
            'CI_High'  : round(ci[2], 2),
        })

        # Bir sonraki ay için lag güncelle
        prev3_usd_fiyat = prev2_usd_fiyat
        prev2_usd_fiyat = prev_usd_fiyat
        prev_usd_fiyat  = nom          # tahmin edilen nominal USD → bir sonraki ayın lag1'i
        prev3_real_usd  = prev_real_usd
        prev_real_usd   = reel

    return pd.DataFrame(results)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Parametreler")
    st.markdown("---")

    st.markdown("### 💱 Kur Ayarları")
    usd_try = st.slider(
        "USD/TRY Kuru (Nisan 2026)",
        min_value=30.0, max_value=60.0, value=44.0, step=0.5,
        help="Nisan 2026 için başlangıç USD/TRY kuru"
    )
    aylik_kur_artis = st.slider(
        "Aylık Kur Değişimi (%)",
        min_value=-2.0, max_value=5.0, value=0.8, step=0.1,
        help="Her ay için beklenen kur artışı (%/ay). 0.8 = %0.8/ay ≈ yıllık %10"
    ) / 100.0

    st.markdown("---")
    st.markdown("### 📊 Görünüm")
    show_ci = st.checkbox("Güven Aralığı Göster", value=True)
    show_historical = st.checkbox("Geçmiş Fiyatları Göster", value=True)

    st.markdown("---")
    st.markdown("### ℹ️ Model Bilgisi")
    st.markdown("""
    **Algoritma:** Weighted Ensemble  
    `XGBoost %72 + Ridge %28`  
    
    **Test MAPE:** %9.05  
    **Test R²:** 0.45  
    
    **Hedef:** Reel USD/kg (2024 baz)  
    **Kapsam:** 152 aylık veri  
    """)

    st.markdown("---")
    st.caption("🌰 Fındık Fiyatı Tahmin Projesi v1.0")


# ─── ANA SAYFA ────────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero-box">
    <p class="hero-title">🌰 Fındık Fiyatı Tahmin Sistemi</p>
    <p class="hero-sub">2026 yılı aylık serbest piyasa fiyat tahminleri · Weighted Ensemble Model · MAPE %9.05</p>
</div>
""", unsafe_allow_html=True)

# Uyarı
st.markdown("""
<div class="warning-card">
⚠️ Bu sistem bir <b>karar destek aracıdır</b>. Tahminler istatistiksel model çıktısıdır; 
spekülatif hareketler, doğal afetler veya ani jeopolitik olayları öngöremez. 
Uzman değerlendirmesiyle birlikte kullanınız.
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Veri ve model yükle ──────────────────────────────────────────────────────
try:
    xgb_m, lgb_m, feat_cols, weights = load_models()
    df = load_history()
    sel_cols, _ = get_feature_cols(df)
except Exception as e:
    st.error(f"Model veya veri yüklenemedi: {e}")
    st.stop()

# ─── 2026 tahminleri ──────────────────────────────────────────────────────────
with st.spinner("2026 tahminleri hesaplanıyor..."):
    pred_df = predict_2026_months(xgb_m, lgb_m, weights, df, sel_cols,
                                   usd_try, aylik_kur_artis)

# ─── ÜST METRİKLER ───────────────────────────────────────────────────────────
son_gercek = df.iloc[-1]
son_tl     = son_gercek.get('Serbest_Piyasa_TL_kg', None)
nisan_tl   = pred_df.iloc[0]['TL']
aralik_tl  = pred_df.iloc[-1]['TL']
yillik_ort = pred_df['TL'].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_nisan = f"+{nisan_tl - son_tl:.1f} TL" if son_tl else None
    st.metric(
        "Nisan 2026 Tahmini",
        f"{nisan_tl:.1f} TL/kg",
        delta=delta_nisan,
        help="Bir sonraki ay tahmini"
    )

with col2:
    st.metric(
        "Aralık 2026 Tahmini",
        f"{aralik_tl:.1f} TL/kg",
        delta=f"{aralik_tl - nisan_tl:+.1f} TL (yılsonu farkı)",
        help="Aralık 2026 tahmini"
    )

with col3:
    st.metric(
        "2026 Yıllık Ortalama",
        f"{yillik_ort:.1f} TL/kg",
        help="Nisan-Aralık ortalaması"
    )

with col4:
    if son_tl:
        st.metric(
            "Son Gerçek Fiyat (Mar-26)",
            f"{son_tl:.1f} TL/kg",
            help="Veri setindeki son gerçek değer"
        )
    else:
        st.metric("USD/TRY Kuru", f"{usd_try:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── ANA GRAFİK ───────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">📈 2026 Aylık Fiyat Tahmini (TL/kg)</p>', unsafe_allow_html=True)

fig = go.Figure()

# Geçmiş fiyatlar (son 24 ay)
if show_historical and 'Serbest_Piyasa_TL_kg' in df.columns:
    hist = df.tail(24).copy()
    fig.add_trace(go.Scatter(
        x=hist['Tarih'],
        y=hist['Serbest_Piyasa_TL_kg'],
        mode='lines+markers',
        name='Gerçek Fiyat (2024-2026)',
        line=dict(color='rgba(255,255,255,0.6)', width=2),
        marker=dict(size=5, color='rgba(255,255,255,0.7)'),
    ))

# Tahmin fiyatları
pred_dates = pd.to_datetime([f"2026-{r['Ay']:02d}-01" for _, r in pred_df.iterrows()])

# Güven aralığı bandı
if show_ci:
    fig.add_trace(go.Scatter(
        x=list(pred_dates) + list(pred_dates[::-1]),
        y=list(pred_df['CI_High']) + list(pred_df['CI_Low'][::-1]),
        fill='toself',
        fillcolor='rgba(124,106,247,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=True,
        name='%90 Güven Aralığı',
        hoverinfo='skip',
    ))

fig.add_trace(go.Scatter(
    x=pred_dates,
    y=pred_df['TL'],
    mode='lines+markers',
    name='2026 Tahmin (TL/kg)',
    line=dict(color='#7c6af7', width=3),
    marker=dict(
        size=9,
        color=pred_df['TL'],
        colorscale='Viridis',
        showscale=False,
        symbol='circle',
        line=dict(color='white', width=1.5)
    ),
    text=[f"{r['Ay_Ad']}: {r['TL']:.1f} TL<br>USD: {r['NomUSD']:.2f}$/kg<br>Kur: {r['Kur']:.2f}"
          for _, r in pred_df.iterrows()],
    hovertemplate='%{text}<extra></extra>',
))

# CI low/high çizgiler (ince kesik)
if show_ci:
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_df['CI_High'],
        mode='lines', line=dict(color='rgba(124,106,247,0.45)', width=1.2, dash='dot'),
        name='CI Üst', showlegend=False, hoverinfo='skip',
    ))
    fig.add_trace(go.Scatter(
        x=pred_dates, y=pred_df['CI_Low'],
        mode='lines', line=dict(color='rgba(124,106,247,0.45)', width=1.2, dash='dot'),
        name='CI Alt', showlegend=False, hoverinfo='skip',
    ))

fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=12),
    hovermode='x unified',
    legend=dict(
        bgcolor='rgba(255,255,255,0.05)',
        bordercolor='rgba(255,255,255,0.1)',
        borderwidth=1,
        x=0.01, y=0.99,
        font=dict(size=11),
    ),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.06)',
        showgrid=True, zeroline=False,
        tickfont=dict(size=11),
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.06)',
        showgrid=True, zeroline=False,
        ticksuffix=' TL',
        tickfont=dict(size=11),
    ),
    height=420,
    margin=dict(l=10, r=10, t=20, b=10),
)

st.plotly_chart(fig, use_container_width=True)

# ─── TABLO ────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

col_tbl, col_senaryo = st.columns([1.1, 0.9])

with col_tbl:
    st.markdown('<p class="section-title">📋 Aylık Tahmin Tablosu</p>', unsafe_allow_html=True)
    display_df = pred_df[['Ay_Ad', 'Kur', 'NomUSD', 'TL', 'CI_Low', 'CI_High']].copy()
    display_df.columns = ['Ay', 'USD/TRY', 'Nominal USD/kg', 'TL/kg Tahmin', 'Alt Sınır (TL)', 'Üst Sınır (TL)']

    def color_tl(val):
        try:
            med = display_df['TL/kg Tahmin'].median()
            if val > med * 1.05:
                return 'color: #ff6b6b; font-weight: 600'
            elif val < med * 0.95:
                return 'color: #6fcf97; font-weight: 600'
            return 'color: #e0e0e0'
        except:
            return ''

    st.dataframe(
        display_df.style
            .format({'USD/TRY': '{:.2f}', 'Nominal USD/kg': '{:.3f}',
                     'TL/kg Tahmin': '{:.1f}', 'Alt Sınır (TL)': '{:.1f}', 'Üst Sınır (TL)': '{:.1f}'})
            .applymap(color_tl, subset=['TL/kg Tahmin'])
            .set_properties(**{
                'background-color': 'rgba(255,255,255,0.02)',
                'border-color': 'rgba(255,255,255,0.08)',
                'font-size': '13px',
            }),
        use_container_width=True,
        height=370,
    )

with col_senaryo:
    st.markdown('<p class="section-title">⚡ Kur Senaryo Analizi (Nisan 2026)</p>', unsafe_allow_html=True)

    senaryo_rows = []
    for pct in [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]:
        kur_s = usd_try * (1 + pct)
        _, _, tl_s = predict_single(xgb_m, lgb_m, weights,
                                     get_feature_cols(df)[1][sel_cols].iloc[-1].to_dict(),
                                     kur_s, 2026)
        senaryo_rows.append({'Kur Değişimi': f"{pct:+.0%}", 'USD/TRY': f"{kur_s:.2f}", 'TL/kg': round(tl_s, 1)})

    sen_df = pd.DataFrame(senaryo_rows)

    fig_s = go.Figure(go.Bar(
        x=sen_df['Kur Değişimi'],
        y=sen_df['TL/kg'],
        marker=dict(
            color=sen_df['TL/kg'],
            colorscale='RdYlGn_r',
            showscale=False,
            line=dict(color='rgba(0,0,0,0)', width=0)
        ),
        text=[f"{v:.0f} TL" for v in sen_df['TL/kg']],
        textposition='outside',
        textfont=dict(color='rgba(255,255,255,0.85)', size=11, family='Inter'),
        hovertemplate='Kur: %{x}<br>Tahmin: %{y:.1f} TL/kg<extra></extra>',
    ))
    fig_s.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=11),
        xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False,
                   ticksuffix=' TL', range=[sen_df['TL/kg'].min()*0.9, sen_df['TL/kg'].max()*1.08]),
        height=350,
        margin=dict(l=5, r=5, t=20, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_s, use_container_width=True)

# ─── NOMİNAL USD GRAFİĞİ ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<p class="section-title">🌍 2026 Reel USD/kg Tahmini (2024 Baz Yılı)</p>', unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=pred_dates,
    y=pred_df['RealUSD'],
    mode='lines+markers',
    name='Reel USD/kg (2024 Baz)',
    line=dict(color='#6fcf97', width=2.5),
    marker=dict(size=8, color='#6fcf97', line=dict(color='white', width=1.5)),
    fill='tozeroy',
    fillcolor='rgba(111,207,151,0.08)',
    hovertemplate='%{x|%B %Y}: %{y:.3f} $/kg<extra></extra>',
))
fig2.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='rgba(255,255,255,0.8)', size=12),
    hovermode='x unified',
    xaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.06)', zeroline=False, ticksuffix=' $/kg'),
    height=280,
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
)
st.plotly_chart(fig2, use_container_width=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:rgba(255,255,255,0.3); font-size:0.78rem'>"
    "🌰 Fındık Fiyatı Tahmin Sistemi · Weighted Ensemble (XGBoost + Ridge) · "
    "Reel USD bazlı tahmin, o dönemin kuru ile TL'ye çevrilir · 2026"
    "</p>",
    unsafe_allow_html=True
)

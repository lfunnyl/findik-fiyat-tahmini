"""
train_model.py
==============
Fındık Fiyatı Tahmin Projesi - Model Eğitim Scripti

Teknik Not (Temporal Covariate Shift Çözümü):
  - Hedef değişken LOG dönüşümü ile normalize edilir (enflasyonist trend baskılanır)
  - Feature selection: Mutual Information bazlı Top-20 özellik seçimi (doğrusal olmayan ilişkiler)
  - Walk-Forward CV: Her fold'da sadece geçmiş veri kullanılır (data leakage yok)
  - Expanding Window: Model her fold'da birikimli geçmiş ile eğitilir

Strateji (5 Adım):
  1. Baseline     : Ridge Regression (referans skor)
  2. Ana Model    : XGBoost (Walk-Forward Expanding Window CV)
  3. Gelişmiş     : LightGBM + SHAP Açıklanabilirlik
  4. CatBoost     : Kategorik uyumlu gradient boosting
  5. Optimizasyon : Optuna Hyperparametre Arama

Çıktı:
  - models/ klasörüne .pkl model dosyaları
  - reports/figures/ klasörüne SHAP ve performans görselleri
  - Konsola detaylı CV + Test skorları
"""

import os
import warnings
import logging
import joblib
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression

import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logging.warning("CatBoost kurulu değil: pip install catboost")
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Konfigürasyon ve Yollar ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(BASE_DIR, "config.yaml"), "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f)

DATA_PATH      = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR     = os.path.join(BASE_DIR, "models")
FIGURES_DIR    = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET         = CFG.get("target", "Fiyat_RealUSD_kg")
TOP_N_FEATURES = int(CFG["model"].get("top_n_features", 15))
FEATURE_METHOD = CFG["model"].get("feature_selection", "vif")
DROP_COLS      = list(CFG.get("drop_cols", []))
VIF_THRESHOLD  = float(CFG.get("multicollinearity", {}).get("vif_threshold", 10.0))


# ─── Yardımcı Fonksiyonlar ──────────────────────────────────────────────────

def load_data():
    logger.info(f"Veri yükleniyor: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih').reset_index(drop=True)
    return df


def prepare_xy(df):
    """
    Feature ve Target matrislerini hazırlar.
    Hedef: log(Y_t) - log(Y_t-1) (Delta Log_Return)
    """
    df = df.copy()
    
    # 1. YENİ YAPI: Sadece Gecikmeli Değişim (Lagged Change) özellikleri eklenecek
    df['USD_MoM_pct']     = df['Fiyat_USD_kg'].shift(1).pct_change(1) * 100
    df['USD_YoY_pct']     = df['Fiyat_USD_kg'].shift(1).pct_change(12) * 100
    df['RealUSD_MoM_pct'] = df['Fiyat_RealUSD_kg'].shift(1).pct_change(1) * 100
    df['RealUSD_YoY_pct'] = df['Fiyat_RealUSD_kg'].shift(1).pct_change(12) * 100
    
    # 2. FEATURE ENGINEERING (SHORT TERM): Hareketli Ortalamalar ve Volatilite
    df['RealUSD_MA3'] = df['Fiyat_RealUSD_kg'].shift(1).rolling(window=3).mean()
    df['RealUSD_MA6'] = df['Fiyat_RealUSD_kg'].shift(1).rolling(window=6).mean()
    
    # Fiyatın MA3'e olan uzaklığı (Momentum/Trend Gücü)
    df['Fiyat_MA3_Farki_Pct'] = (df['Fiyat_RealUSD_kg'].shift(1) - df['RealUSD_MA3']) / df['RealUSD_MA3'] * 100
    
    # Kur aylık ivmesi ve Kur Volatilitesi
    df['Kur_Aylik_Ivme']  = df['USD_TRY_Kapanis'].shift(1).pct_change(1) * 100
    df['Kur_Volatilite_3Ay'] = df['USD_TRY_Kapanis'].shift(1).rolling(window=3).std()
    
    df = df.bfill().ffill()
    
    # --- YENİ EKLENEN: Regime Detection (Şok Alarmı) ---
    volatilite_mean = df['Kur_Volatilite_3Ay'].mean()
    # Kur volatilitesi normalin 2 katından fazlaysa VEYA dondan etkilenmişse 1 (Şok Rejimi)
    is_shock = (df['Kur_Volatilite_3Ay'] > volatilite_mean * 2)
    if 'Kritik_Don' in df.columns:
        is_shock = is_shock | (df['Kritik_Don'] > 0)
    df['Regime_Shock_Warning'] = np.where(is_shock, 1, 0)

    # --- YENİ EKLENEN: TMO Müdahalesi (Policy Causal Feature) ---
    # TMO fiyatı devlet tarafından sezon başında açıklanır (Dışsaldır, sızıntı yaratmaz).
    # TMO fiyatındaki o ayki artış:
    df['TMO_Fiyat_Artis_Pct'] = df['TMO_Giresun_TL_kg'].pct_change(1) * 100
    
    # TMO'nun açıkladığı güncel fiyatın, GEÇEN AYKİ serbest piyasaya göre farkı (Makas)
    # Makas devasa pozitifse (+%50), serbest piyasa o ay fırlamak zorundadır!
    df['TMO_Mevcut_Makas_Pct'] = (df['TMO_Giresun_TL_kg'] - df['Serbest_Piyasa_TL_kg'].shift(1)) / df['Serbest_Piyasa_TL_kg'].shift(1) * 100

    # 3. YASAKLI LİSTE: Mutlak geçmiş fiyatları modele GÖSTERMEYİZ
    yasakli_laglar = [
        "Fiyat_Lag1", "Fiyat_Lag2", "Fiyat_Lag3", "Fiyat_Lag12",
        "USD_Lag1", "USD_Lag2", "USD_Lag3", "USD_Lag12",
        "RealUSD_Lag1", "RealUSD_Lag3"
    ]
    
    # TARGET HESAPLAMA (DELTA)
    y_raw = df[TARGET]
    y_log = np.log1p(y_raw)
    y_log_prev = y_log.shift(1)
    y_log_diff = y_log - y_log_prev
    
    # NaN olan ilk satırı düş
    df['y_log_diff'] = y_log_diff
    df['y_log_prev'] = y_log_prev
    valid_idx = df['y_log_diff'].notna()
    
    df = df[valid_idx].copy()
    y_log_diff = df['y_log_diff']
    y_log_prev = df['y_log_prev']
    y_raw = df[TARGET]
    
    drop_existing = [c for c in DROP_COLS + yasakli_laglar if c in df.columns]
    X_raw = df.drop(columns=drop_existing + ['y_log_diff', 'y_log_prev']).select_dtypes(include=[np.number])
    
    return X_raw, y_log_diff, y_log_prev, y_raw


def select_features(X_train, y_train, top_n=TOP_N_FEATURES, method=FEATURE_METHOD):
    """
    Ağaç bazlı modeller (XGBoost/LightGBM/CatBoost) için feature selection.
    Bu modeller multikollineariteyi zaten doğal olarak handle eder.
    VIF burada UYGULANMAZ — o sadece Ridge için (select_features_ridge).
    """
    if method in ('mutual_info', 'vif'):  # vif modunda bile tree modeller MI kullansin
        mi_scores = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=5)
        mi_series = pd.Series(mi_scores, index=X_train.columns).dropna()
        selected  = mi_series.nlargest(top_n).index.tolist()
    else:
        corr = X_train.corrwith(y_train).abs().dropna()
        selected = corr.nlargest(top_n).index.tolist()

    # --- YENİ EKLENEN: Causal Forcing (Nedensel Zorlama) ---
    # Şokları tetikleyen asıl değişkenler "nadir" oldukları için MI testinden elenebiliyor.
    # Bunları modele ZORLA sokacağız.
    vip_features = ['Kritik_Don', 'Rekolte_Surprise_Pct', 'Kur_Volatilite_3Ay', 'Regime_Shock_Warning', 'TMO_Mevcut_Makas_Pct', 'TMO_Fiyat_Artis_Pct']
    vip_to_add = [f for f in vip_features if f in X_train.columns and f not in selected]
    
    if vip_to_add:
        # Eğer varsa, en sondaki önemsizleri çıkarıp VIP'leri ekleyelim (boyut aynı kalsın)
        selected = selected[:top_n - len(vip_to_add)] + vip_to_add
        logger.info(f"  [CAUSAL FORCING] {vip_to_add} modele ZORLA eklendi.")

    logger.info(f"  Feature Selection: {len(X_train.columns)} → {len(selected)} özellik")
    logger.info(f"  Top-5: {selected[:5]}")
    return selected


def select_features_ridge(X_train, y_train, top_n=TOP_N_FEATURES):
    """
    Sadece Ridge Regression için: MI + VIF hibrit seçimi.
    En önemli top-3 MI feature'ları (RealUSD_Lag1 vb.) her zaman korunur,
    geri kalan adaylar arasından VIF > threshold olan elenir.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    mi_scores  = mutual_info_regression(X_train, y_train, random_state=42, n_neighbors=5)
    mi_series  = pd.Series(mi_scores, index=X_train.columns).dropna()
    top_by_mi  = mi_series.nlargest(top_n).index.tolist()

    X_sub     = X_train[top_by_mi].fillna(0).copy()
    protected = set(top_by_mi[:3])   # En kritik 3 feature her zaman kalsın

    while X_sub.shape[1] > 5:
        vifs       = [variance_inflation_factor(X_sub.values, i) for i in range(X_sub.shape[1])]
        vif_series = pd.Series(vifs, index=X_sub.columns)
        candidates = vif_series.drop(labels=[c for c in protected if c in vif_series.index])
        if candidates.empty or candidates.max() <= VIF_THRESHOLD:
            break
        X_sub = X_sub.drop(columns=[candidates.idxmax()])

    selected = X_sub.columns.tolist()
    logger.info(f"  Feature Selection (MI+VIF Ridge): {len(X_train.columns)} → {len(selected)} özellik")
    return selected


def metrics_log(y_true, y_pred, prefix=""):
    """Orijinal ölçekteki tahminleri (USD bazında) alır, metrikleri hesapla."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    logger.info(f"{prefix} → MAE: {mae:.3f} USD/kg | RMSE: {rmse:.3f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'y_pred_orig': y_pred}


def walk_forward_expanding_cv(model_factory, X, y_log_diff, y_log_prev, n_splits=5, model_name="Model"):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    all_val_true, all_val_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr  = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr  = y_log_diff.iloc[train_idx]
        y_val_diff = y_log_diff.iloc[val_idx]
        y_val_prev = y_log_prev.iloc[val_idx]

        sel_cols  = select_features(X_tr, y_tr, top_n=TOP_N_FEATURES)
        X_tr_sel  = X_tr[sel_cols]
        X_val_sel = X_val[sel_cols]

        model = model_factory()
        model.fit(X_tr_sel, y_tr)
        preds_log_diff = model.predict(X_val_sel)

        # Delta Modeling Reconstruction
        y_val_orig = np.expm1(y_val_prev.values + y_val_diff.values)
        preds_orig = np.expm1(y_val_prev.values + preds_log_diff)

        r2  = r2_score(y_val_orig, preds_orig)
        mae = mean_absolute_error(y_val_orig, preds_orig)
        cv_scores.append({'Fold': fold, 'R2': r2, 'MAE': mae, 'train_size': len(train_idx)})
        logger.info(
            f"  [{model_name}] Fold {fold} (train={len(train_idx)}, val={len(val_idx)}) "
            f"→ R²: {r2:.4f} | MAE: {mae:.2f} TL"
        )
        all_val_true.extend(y_val_orig.tolist())
        all_val_pred.extend(preds_orig.tolist())

    avg_r2    = np.mean([s['R2']  for s in cv_scores])
    avg_mae   = np.mean([s['MAE'] for s in cv_scores])
    overall_r2 = r2_score(all_val_true, all_val_pred)
    logger.info(f"  [{model_name}] CV Ortalama (fold bazlı)  → R²: {avg_r2:.4f} | MAE: {avg_mae:.2f} TL")
    logger.info(f"  [{model_name}] Tüm CV Verisi Birleşik R²: {overall_r2:.4f}")
    return cv_scores


def plot_predictions(y_test_orig, preds_dict, dates_test):
    plt.figure(figsize=(14, 5))
    plt.plot(dates_test.values, y_test_orig.values, label='Gerçek Fiyat (Reel USD)', color='black', lw=2.5)
    colors = ['royalblue', 'tomato', 'seagreen', 'mediumpurple', 'darkorange']
    for (name, preds), color in zip(preds_dict.items(), colors):
        plt.plot(dates_test.values, preds, label=name, linestyle='--', color=color, lw=1.8, alpha=0.85)
    plt.title('Model Tahminleri vs Gerçek Fındık Fiyatı (Test Seti - Reel USD/kg, 2024 Baz)', fontsize=13)
    plt.xlabel('Tarih')
    plt.ylabel('Reel USD/kg (2024 Baz Yılı)')
    plt.legend()
    plt.tight_layout()
    fpath = os.path.join(FIGURES_DIR, '06_model_tahminleri.png')
    plt.savefig(fpath, dpi=300)
    plt.close()
    logger.info(f"Model tahmin grafiği kaydedildi → {fpath}")


def plot_feature_importance(model, feature_names, model_name, top_n=20):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names).nlargest(top_n)
        plt.figure(figsize=(10, 6))
        imp[::-1].plot(kind='barh', color='steelblue')
        plt.title(f'{model_name} - En Önemli {top_n} Özellik')
        plt.tight_layout()
        fname = f'07_feature_importance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300)
        plt.close()
        logger.info(f"Feature Importance grafiği kaydedildi: {fname}")


def plot_shap(model, X_test_sel, model_name):
    logger.info(f"SHAP değerleri hesaplanıyor ({model_name})...")
    try:
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X_test_sel)
        plt.figure()
        shap.summary_plot(shap_vals, X_test_sel, show=False, max_display=20)
        plt.title(f'SHAP Özet Grafiği - {model_name}')
        plt.tight_layout()
        safe = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        fname = f'08_shap_{safe}.png'
        plt.savefig(os.path.join(FIGURES_DIR, fname), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"SHAP grafiği kaydedildi: {fname}")
    except Exception as e:
        logger.warning(f"SHAP hesaplanamadı: {e}")


# ─── ADIM 1: Baseline (Ridge Regression) ────────────────────────────────────

def train_baseline(X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 1: BASELINE — Ridge Regression (Delta Modeling)")
    logger.info("="*60)
    sel_cols = select_features_ridge(X_train, y_train_log_diff)
    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr_sel)
    X_te_s    = scaler.transform(X_te_sel)

    ridge     = Ridge(alpha=10.0)
    ridge.fit(X_tr_s, y_train_log_diff)
    preds_log_diff = ridge.predict(X_te_s)

    preds_orig = np.expm1(y_test_log_prev.values + preds_log_diff)
    scores = metrics_log(y_test_raw.values, preds_orig, prefix="Ridge Baseline (Test)")
    joblib.dump({'model': ridge, 'features': sel_cols, 'scaler': scaler}, os.path.join(MODELS_DIR, 'ridge_model.pkl'))
    return scores['y_pred_orig'], scores



# ─── ADIM 2: XGBoost ────────────────────────────────────────────────────────

def train_xgboost(X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 2: ANA MODEL — XGBoost (Delta Modeling)")
    logger.info("="*60)

    def xgb_factory():
        return xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_alpha=0.3, reg_lambda=1.0,
            random_state=42, verbosity=0
        )

    walk_forward_expanding_cv(xgb_factory, X, y_log_diff, y_log_prev, n_splits=5, model_name="XGBoost")

    sel_cols = select_features(X_train, y_train_log_diff)
    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_alpha=0.3, reg_lambda=1.0,
        random_state=42, verbosity=0, early_stopping_rounds=30
    )
    xgb_model.fit(
        X_tr_sel, y_train_log_diff,
        eval_set=[(X_te_sel, y_test_log_diff)],
        verbose=False
    )
    preds_log_diff = xgb_model.predict(X_te_sel)
    preds_orig = np.expm1(y_test_log_prev.values + preds_log_diff)
    
    scores = metrics_log(y_test_raw.values, preds_orig, prefix="XGBoost (Test)")
    joblib.dump({'model': xgb_model, 'features': sel_cols}, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    plot_feature_importance(xgb_model, sel_cols, "XGBoost")
    return xgb_model, sel_cols, scores['y_pred_orig'], scores


# ─── ADIM 3: LightGBM + SHAP ────────────────────────────────────────────────

def train_lightgbm(X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 3: GELİŞMİŞ — LightGBM + SHAP (Delta Modeling)")
    logger.info("="*60)

    def lgb_factory():
        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=5,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1
        )

    walk_forward_expanding_cv(lgb_factory, X, y_log_diff, y_log_prev, n_splits=5, model_name="LightGBM")

    sel_cols = select_features(X_train, y_train_log_diff)
    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31, max_depth=5,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1
    )
    lgb_model.fit(X_tr_sel, y_train_log_diff)
    preds_log_diff = lgb_model.predict(X_te_sel)
    preds_orig = np.expm1(y_test_log_prev.values + preds_log_diff)
    
    scores = metrics_log(y_test_raw.values, preds_orig, prefix="LightGBM (Test)")
    joblib.dump({'model': lgb_model, 'features': sel_cols}, os.path.join(MODELS_DIR, 'lightgbm_model.pkl'))
    plot_shap(lgb_model, X_te_sel, "LightGBM")
    return lgb_model, sel_cols, scores['y_pred_orig'], scores


# ─── ADIM 4: CatBoost ────────────────────────────────────────────────────

def train_catboost(X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 4: ROBUST — CatBoost (Delta Modeling)")
    logger.info("="*60)

    def cat_factory():
        from catboost import CatBoostRegressor
        return CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=5,
            l2_leaf_reg=3.0, random_seed=42, verbose=False
        )

    try:
        from catboost import CatBoostRegressor
        walk_forward_expanding_cv(cat_factory, X, y_log_diff, y_log_prev, n_splits=5, model_name="CatBoost")

        sel_cols = select_features(X_train, y_train_log_diff)
        X_tr_sel = X_train[sel_cols]
        X_te_sel = X_test[sel_cols]

        cat_model = CatBoostRegressor(
            iterations=300, learning_rate=0.05, depth=5,
            l2_leaf_reg=3.0, random_seed=42, verbose=False, early_stopping_rounds=30
        )
        cat_model.fit(X_tr_sel, y_train_log_diff, eval_set=(X_te_sel, y_test_log_diff), verbose=False)
        preds_log_diff = cat_model.predict(X_te_sel)
        preds_orig = np.expm1(y_test_log_prev.values + preds_log_diff)
        
        scores = metrics_log(y_test_raw.values, preds_orig, prefix="CatBoost (Test)")
        joblib.dump({'model': cat_model, 'features': sel_cols}, os.path.join(MODELS_DIR, 'catboost_model.pkl'))
        return cat_model, sel_cols, scores['y_pred_orig'], scores
    except ImportError:
        logger.warning("CatBoost yüklü değil, atlanıyor.")
        return None, None, None, None

    logger.info("\n" + "="*60)
    logger.info("ADIM 4: CatBoost (Gradient Boosting)")
    logger.info("="*60)

    def cat_factory():
        return CatBoostRegressor(
            iterations=500, learning_rate=0.05, depth=4,
            l2_leaf_reg=3.0, random_seed=42, verbose=0,
            loss_function='RMSE'
        )

    walk_forward_expanding_cv(cat_factory, X, y_log, n_splits=5, model_name="CatBoost")

    sel_cols = select_features(X_train, y_train_log)
    X_tr_sel, X_te_sel = X_train[sel_cols], X_test[sel_cols]

    cat_model = CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=4,
        l2_leaf_reg=3.0, random_seed=42, verbose=0,
        loss_function='RMSE'
    )
    cat_model.fit(X_tr_sel, y_train_log, eval_set=(X_te_sel, np.log1p(y_test_raw)),
                  early_stopping_rounds=30)
    preds_log = cat_model.predict(X_te_sel)
    scores = metrics_log(np.log1p(y_test_raw), preds_log, prefix="CatBoost (Test)")
    joblib.dump({'model': cat_model, 'features': sel_cols}, os.path.join(MODELS_DIR, 'catboost_model.pkl'))
    plot_feature_importance(cat_model, sel_cols, "CatBoost")
    return cat_model, sel_cols, scores['y_pred_orig'], scores


# ─── ADIM 4: Optuna ─────────────────────────────────────────────────────────

def optuna_optimize(X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw, sel_cols, n_trials=50):
    logger.info("\n" + "="*60)
    logger.info("ADIM 5: OPTUNA — Hiperparametre Optimizasyonu (Delta Modeling)")
    logger.info("="*60)

    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    def objective_xgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'random_state': 42,
            'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X_tr_sel, y_train_log_diff, scoring='neg_mean_squared_error', cv=TimeSeriesSplit(n_splits=3))
        return np.mean(scores)

    def objective_lgb(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'random_state': 42,
            'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(model, X_tr_sel, y_train_log_diff, scoring='neg_mean_squared_error', cv=TimeSeriesSplit(n_splits=3))
        return np.mean(scores)

    logger.info("XGBoost Optuna çalışıyor...")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgb, n_trials=n_trials)
    
    logger.info("LightGBM Optuna çalışıyor...")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=n_trials)

    best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
    best_xgb.fit(X_tr_sel, y_train_log_diff)
    preds_xgb_diff = best_xgb.predict(X_te_sel)
    preds_xgb_orig = np.expm1(y_test_log_prev.values + preds_xgb_diff)
    sc_xgb = metrics_log(y_test_raw.values, preds_xgb_orig, prefix="Best XGBoost (Optuna)")

    best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)
    best_lgb.fit(X_tr_sel, y_train_log_diff)
    preds_lgb_diff = best_lgb.predict(X_te_sel)
    preds_lgb_orig = np.expm1(y_test_log_prev.values + preds_lgb_diff)
    sc_lgb = metrics_log(y_test_raw.values, preds_lgb_orig, prefix="Best LightGBM (Optuna)")

    return sc_xgb['y_pred_orig'], sc_xgb, sc_lgb['y_pred_orig'], sc_lgb


# ─── ÖZET TABLO ─────────────────────────────────────────────────────────────

def print_summary(results: dict):
    logger.info("\n" + "="*68)
    logger.info("📊 SONUÇ ÖZETI — TÜM MODELLER (Test Seti, Reel USD/kg 2024 Bazı)")
    logger.info("="*68)
    logger.info(f"{'Model':<30} {'R²':>8} {'MAE (USD)':>12} {'RMSE':>10} {'MAPE%':>8}")
    logger.info("-"*68)
    for name, sc in results.items():
        logger.info(f"{name:<30} {sc['R2']:>8.4f} {sc['MAE']:>12.4f} {sc['RMSE']:>10.4f} {sc['MAPE']:>8.2f}")
    logger.info("="*68)


# ─── ANA AKIŞ ───────────────────────────────────────────────────────────────

def main():
    df = load_data()
    X, y_log_diff, y_log_prev, y_raw = prepare_xy(df)

    split_idx   = int(len(X) * 0.80)
    X_train     = X.iloc[:split_idx]
    X_test      = X.iloc[split_idx:]
    y_train_log_diff = y_log_diff.iloc[:split_idx]
    y_test_log_diff  = y_log_diff.iloc[split_idx:]
    y_test_log_prev  = y_log_prev.iloc[split_idx:]
    y_test_raw  = y_raw.iloc[split_idx:]
    
    # df has valid_idx already applied in prepare_xy, so indexing matches
    # The dates might be misaligned if df inside prepare_xy drops rows but original df doesn't.
    # Actually dates are not dropped in original df. 
    # Let's get dates from the subset.
    # Wait! If prepare_xy drops 1 row, df has 1 less row than original df. 
    # Let's fix this inside main by using the df index.
    dates_test = df['Tarih'].iloc[1:].iloc[split_idx:]

    logger.info(f"Train: {len(X_train)} ay")
    logger.info(f"Test : {len(X_test)} ay")

    all_scores = {}
    all_preds  = {}

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Findik_Fiyat_Modelleri")

    with mlflow.start_run(run_name="Ridge_Baseline"):
        preds_ridge, sc_ridge = train_baseline(X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw)
        all_scores['Ridge Baseline'] = sc_ridge
        all_preds['Ridge Baseline']  = preds_ridge

    with mlflow.start_run(run_name="XGBoost_Default"):
        xgb_model, xgb_cols, preds_xgb, sc_xgb = train_xgboost(
            X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_diff, y_test_log_prev, y_test_raw
        )
        all_scores['XGBoost'] = sc_xgb
        all_preds['XGBoost']  = preds_xgb

    with mlflow.start_run(run_name="LightGBM_Default"):
        lgb_model, lgb_cols, preds_lgb, sc_lgb = train_lightgbm(
            X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw
        )
        all_scores['LightGBM'] = sc_lgb
        all_preds['LightGBM']  = preds_lgb

    with mlflow.start_run(run_name="CatBoost_Default"):
        cat_model, cat_cols, preds_cat, sc_cat = train_catboost(
            X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_diff, y_test_log_prev, y_test_raw
        )
        if sc_cat:
            all_scores['CatBoost'] = sc_cat
            all_preds['CatBoost']  = preds_cat

    with mlflow.start_run(run_name="Optuna_Best_Models"):
        preds_xgb_o, sc_xgb_o, preds_lgb_o, sc_lgb_o = optuna_optimize(
            X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw, xgb_cols, n_trials=50
        )
        all_scores['XGBoost (Optuna)']  = sc_xgb_o
        all_scores['LightGBM (Optuna)'] = sc_lgb_o
        all_preds['XGBoost (Optuna)']   = preds_xgb_o
        all_preds['LightGBM (Optuna)']  = preds_lgb_o

    # Tahmin Grafiği
    plot_predictions(y_test_raw, all_preds, dates_test)

    # Şok Dönemi MAPE Hesaplama (Stress Testi)
    # y_test_raw ve y_test_lag1 (expm1(y_test_log_prev)) kullanarak şokları bul
    y_test_lag1_orig = np.expm1(y_test_log_prev.values)
    fiyat_degisimi = np.abs((y_test_raw.values - y_test_lag1_orig) / y_test_lag1_orig) * 100
    shock_mask = fiyat_degisimi > 10.0
    
    if shock_mask.sum() > 0:
        logger.info(f"\n[STRESS TEST] Test setinde {shock_mask.sum()} adet Şok Dönemi (>%10 değişim) bulundu.")
        for name, sc in all_scores.items():
            preds = np.array(all_preds[name])
            y_test_shock = y_test_raw.values[shock_mask]
            preds_shock = preds[shock_mask]
            mape_shock = np.mean(np.abs((y_test_shock - preds_shock) / np.where(y_test_shock==0, 1, y_test_shock))) * 100
            sc['Shock_MAPE'] = float(mape_shock)
    else:
        for name, sc in all_scores.items():
            sc['Shock_MAPE'] = sc['MAPE']

    # Özet
    print_summary(all_scores)

    # Tüm sonuçları JSON olarak kaydet
    import json
    scores_serializable = {
        k: {m: float(v) for m, v in sc.items() if m != 'y_pred_orig'}
        for k, sc in all_scores.items()
    }
    json_path = os.path.join(MODELS_DIR, 'all_model_scores.json')
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(scores_serializable, jf, indent=2, ensure_ascii=False)
    logger.info(f"Tüm model skorları kaydedildi → {json_path}")


if __name__ == "__main__":
    main()

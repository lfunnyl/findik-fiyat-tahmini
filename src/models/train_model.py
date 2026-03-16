"""
train_model.py
==============
Fındık Fiyatı Tahmin Projesi - Model Eğitim Scripti

Teknik Not (Temporal Covariate Shift Çözümü):
  - Hedef değişken LOG dönüşümü ile normalize edilir (enflasyonist trend baskılanır)
  - Feature selection: Korelasyon bazlı Top-30 özellik seçimi (overfitting azaltılır)
  - Walk-Forward CV: Her fold'da sadece geçmiş veri kullanılır (data leakage yok)
  - Expanding Window: Model her fold'da birikimli geçmiş ile eğitilir

Strateji (4 Adım):
  1. Baseline     : Ridge Regression (referans skor)
  2. Ana Model    : XGBoost (Walk-Forward Expanding Window CV)
  3. Gelişmiş     : LightGBM + SHAP Açıklanabilirlik
  4. Optimizasyon : Optuna Hyperparametre Arama

Çıktı:
  - models/ klasörüne .pkl model dosyaları
  - reports/figures/ klasörüne SHAP ve performans görselleri
  - Konsola detaylı CV + Test skorları
"""

import os
import warnings
import logging
import joblib
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Sabit Yollar ───────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET = 'Fiyat_RealUSD_kg'  # 2024 baz yıllı reel USD (ABD enflasyonundan arındırılmış)

# Model eğitiminde kullanmayacağımız sütunlar
DROP_COLS = [
    TARGET,
    # Diğer hedef / türetilmiş fiyat sütunları (data leakage)
    'Serbest_Piyasa_TL_kg', 'Fiyat_USD_kg', 'US_CPI_Carpani',
    # Kimlik / zaman sütunları
    'Tarih', 'Yil_Ay', 'Hasat_Donemi',
    # Hedefle 0.98+ korelasyon → data leakage
    'TMO_Giresun_TL_kg', 'TMO_Levant_TL_kg',
    # Fiyat lag'ıları → dağılım kayması
    'Fiyat_Lag1', 'Fiyat_Lag2', 'Fiyat_Lag3', 'Fiyat_Lag12',
    # Hedeften türetilmiş değişim yüzdeleri
    'Fiyat_Degisim_1A_Pct', 'Fiyat_Degisim_3A_Pct',
    # Hedeften türetilmiş oran
    'Fiyat_bolu_AsgariUcret_Orani',
    # Saf trend proxy’sı
    'Yil', 'Sezon_Yili',
]

TOP_N_FEATURES = 20  # Korelasyon bazı feature selection



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
    - Hedef: log1p(Fiyat_RealUSD_kg) → USD enflasyonundan arındırılmış, scale-invariant
    - USD bazlı lag'lar eklenir: TL lag'larından daha az temporal shift yaşar
    - Sadece sayısal sütunlar kullanılır
    """
    df = df.copy()
    # USD bazlı lag ve momentum özellikleri (TL'ye kıyasla daha stabil ölçek)
    df['USD_Lag1']      = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']      = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']      = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']     = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']   = df['Fiyat_USD_kg'].pct_change(1) * 100   # Aylık değişim %
    df['USD_YoY_pct']   = df['Fiyat_USD_kg'].pct_change(12) * 100  # Yıllık değişim %
    # RealUSD bazlı lag (USD + CPI etkisi birlikte)
    df['RealUSD_Lag1']  = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3']  = df['Fiyat_RealUSD_kg'].shift(3)
    # Eksikleri doldur (shift'ten kaynaklanan başlangıç NaN'lar)
    df = df.bfill().ffill()

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X_raw = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    y_raw = df[TARGET]
    y_log = np.log1p(y_raw)
    return X_raw, y_log, y_raw


def select_features(X_train, y_train, top_n=TOP_N_FEATURES):
    """
    Train seti üzerinde korelasyon bazlı feature selection.
    Yalnızca train verisini kullanarak seçim yapılır (test sızıntısı yok).
    """
    corr = X_train.corrwith(y_train).abs().dropna()
    selected = corr.nlargest(top_n).index.tolist()
    logger.info(f"  Feature Selection: {len(X_train.columns)} → {len(selected)} özellik")
    logger.info(f"  Top 10: {selected[:10]}")
    return selected


def metrics_log(y_true_log, y_pred_log, prefix=""):
    """Log uzayındaki tahminleri orijinal ölçeğe çevir (USD bazında), metrikleri hesapla."""
    y_true = np.expm1(np.asarray(y_true_log))
    y_pred = np.expm1(np.asarray(y_pred_log))
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    logger.info(f"{prefix} → MAE: {mae:.3f} USD/kg | RMSE: {rmse:.3f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'y_pred_orig': y_pred}


def walk_forward_expanding_cv(model_factory, X, y_log, n_splits=5, model_name="Model"):
    """
    Expanding Window Walk-Forward CV.
    TimeSeriesSplit doğal olarak expanding window'dur:
    Her fold'da birikimli geçmiş (train) → ileriki dönem (val).
    Feature selection her fold'da ayrı yapılır (fold-wise, leakage-free).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    all_val_true, all_val_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr  = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_tr  = y_log.iloc[train_idx]
        y_val = y_log.iloc[val_idx]

        # Her fold'da kendi train seti üzerinde feature selection
        sel_cols  = select_features(X_tr, y_tr, top_n=TOP_N_FEATURES)
        X_tr_sel  = X_tr[sel_cols]
        X_val_sel = X_val[sel_cols]

        model = model_factory()
        model.fit(X_tr_sel, y_tr)
        preds_log = model.predict(X_val_sel)

        y_val_orig = np.expm1(y_val.values)
        preds_orig = np.expm1(preds_log)

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

def train_baseline(X_train, X_test, y_train_log, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 1: BASELINE — Ridge Regression (Log-Transform)")
    logger.info("="*60)
    sel_cols = select_features(X_train, y_train_log)
    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    scaler    = StandardScaler()
    X_tr_s    = scaler.fit_transform(X_tr_sel)
    X_te_s    = scaler.transform(X_te_sel)

    ridge     = Ridge(alpha=10.0)
    ridge.fit(X_tr_s, y_train_log)
    preds_log = ridge.predict(X_te_s)

    scores = metrics_log(np.log1p(y_test_raw), preds_log, prefix="Ridge Baseline (Test)")
    joblib.dump({'model': ridge, 'features': sel_cols, 'scaler': scaler}, os.path.join(MODELS_DIR, 'ridge_model.pkl'))
    return scores['y_pred_orig'], scores



# ─── ADIM 2: XGBoost ────────────────────────────────────────────────────────

def train_xgboost(X, y_log, X_train, X_test, y_train_log, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 2: ANA MODEL — XGBoost (Expanding Window CV)")
    logger.info("="*60)

    def xgb_factory():
        return xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_alpha=0.3, reg_lambda=1.0,
            random_state=42, verbosity=0
        )

    walk_forward_expanding_cv(xgb_factory, X, y_log, n_splits=5, model_name="XGBoost")

    # Final model
    sel_cols = select_features(X_train, y_train_log)
    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    xgb_model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.7,
        min_child_weight=5, reg_alpha=0.3, reg_lambda=1.0,
        random_state=42, verbosity=0, early_stopping_rounds=30
    )
    xgb_model.fit(
        X_tr_sel, y_train_log,
        eval_set=[(X_te_sel, np.log1p(y_test_raw))],
        verbose=False
    )
    preds_log = xgb_model.predict(X_te_sel)
    scores = metrics_log(np.log1p(y_test_raw), preds_log, prefix="XGBoost (Test)")
    joblib.dump({'model': xgb_model, 'features': sel_cols}, os.path.join(MODELS_DIR, 'xgboost_model.pkl'))
    plot_feature_importance(xgb_model, sel_cols, "XGBoost")
    return xgb_model, sel_cols, scores['y_pred_orig'], scores


# ─── ADIM 3: LightGBM + SHAP ────────────────────────────────────────────────

def train_lightgbm(X, y_log, X_train, X_test, y_train_log, y_test_raw):
    logger.info("\n" + "="*60)
    logger.info("ADIM 3: GELİŞMİŞ — LightGBM + SHAP")
    logger.info("="*60)

    def lgb_factory():
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=20,
            min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1
        )

    walk_forward_expanding_cv(lgb_factory, X, y_log, n_splits=5, model_name="LightGBM")

    sel_cols = select_features(X_train, y_train_log)
    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    lgb_model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=20,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1
    )
    lgb_model.fit(X_tr_sel, y_train_log)
    preds_log = lgb_model.predict(X_te_sel)
    scores = metrics_log(np.log1p(y_test_raw), preds_log, prefix="LightGBM (Test)")
    joblib.dump({'model': lgb_model, 'features': sel_cols}, os.path.join(MODELS_DIR, 'lightgbm_model.pkl'))
    plot_feature_importance(lgb_model, sel_cols, "LightGBM")
    plot_shap(lgb_model, X_te_sel, "LightGBM")
    return lgb_model, sel_cols, scores['y_pred_orig'], scores


# ─── ADIM 4: Optuna ─────────────────────────────────────────────────────────

def optuna_optimize(X_train, X_test, y_train_log, y_test_raw, sel_cols, n_trials=50):
    logger.info("\n" + "="*60)
    logger.info(f"ADIM 4: OPTİMİZASYON — Optuna ({n_trials} trial)")
    logger.info("="*60)

    X_tr_sel = X_train[sel_cols]
    X_te_sel = X_test[sel_cols]

    # XGBoost objective
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'random_state': 42, 'verbosity': 0
        }
        model = xgb.XGBRegressor(**params)
        tscv  = TimeSeriesSplit(n_splits=5)
        maes  = []
        for tr_idx, val_idx in tscv.split(X_tr_sel):
            model.fit(X_tr_sel.iloc[tr_idx], y_train_log.iloc[tr_idx])
            preds_orig = np.expm1(model.predict(X_tr_sel.iloc[val_idx]))
            true_orig  = np.expm1(y_train_log.iloc[val_idx].values)
            maes.append(mean_absolute_error(true_orig, preds_orig))
        return np.mean(maes)

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(xgb_objective, n_trials=n_trials // 2, show_progress_bar=False)
    logger.info(f"  XGBoost Optuna → En iyi CV MAE: {study_xgb.best_value:.2f} TL")

    best_xgb = xgb.XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
    best_xgb.fit(X_tr_sel, y_train_log)
    sc_xgb = metrics_log(np.log1p(y_test_raw), best_xgb.predict(X_te_sel), prefix="XGBoost Optuna (Test)")
    joblib.dump({'model': best_xgb, 'features': sel_cols}, os.path.join(MODELS_DIR, 'xgboost_optuna_model.pkl'))

    # LightGBM objective
    def lgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'num_leaves': trial.suggest_int('num_leaves', 10, 60),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 40),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 3.0),
            'random_state': 42, 'verbose': -1
        }
        model = lgb.LGBMRegressor(**params)
        tscv  = TimeSeriesSplit(n_splits=5)
        maes  = []
        for tr_idx, val_idx in tscv.split(X_tr_sel):
            model.fit(X_tr_sel.iloc[tr_idx], y_train_log.iloc[tr_idx])
            preds_orig = np.expm1(model.predict(X_tr_sel.iloc[val_idx]))
            true_orig  = np.expm1(y_train_log.iloc[val_idx].values)
            maes.append(mean_absolute_error(true_orig, preds_orig))
        return np.mean(maes)

    study_lgb = optuna.create_study(direction='minimize')
    study_lgb.optimize(lgb_objective, n_trials=n_trials // 2, show_progress_bar=False)
    logger.info(f"  LightGBM Optuna → En iyi CV MAE: {study_lgb.best_value:.2f} TL")

    best_lgb = lgb.LGBMRegressor(**study_lgb.best_params, random_state=42, verbose=-1)
    best_lgb.fit(X_tr_sel, y_train_log)
    sc_lgb = metrics_log(np.log1p(y_test_raw), best_lgb.predict(X_te_sel), prefix="LightGBM Optuna (Test)")
    joblib.dump({'model': best_lgb, 'features': sel_cols}, os.path.join(MODELS_DIR, 'lightgbm_optuna_model.pkl'))

    # SHAP en iyi Optuna modeline uygula
    best_name  = "XGBoost Optuna" if sc_xgb['MAE'] < sc_lgb['MAE'] else "LightGBM Optuna"
    best_model = best_xgb          if sc_xgb['MAE'] < sc_lgb['MAE'] else best_lgb
    logger.info(f"\n  🏆 Optuna En İyi: {best_name} (MAE: {min(sc_xgb['MAE'], sc_lgb['MAE']):.2f} TL)")
    plot_shap(best_model, X_te_sel, best_name)

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
    X, y_log, y_raw = prepare_xy(df)

    # Son %20 = Test (~30 ay)
    split_idx   = int(len(df) * 0.80)
    X_train     = X.iloc[:split_idx]
    X_test      = X.iloc[split_idx:]
    y_train_log = y_log.iloc[:split_idx]
    y_test_raw  = y_raw.iloc[split_idx:]
    dates_test  = df['Tarih'].iloc[split_idx:]

    logger.info(
        f"Train: {len(X_train)} ay "
        f"({df['Tarih'].iloc[0].strftime('%Y-%m')} → {df['Tarih'].iloc[split_idx-1].strftime('%Y-%m')})"
    )
    logger.info(
        f"Test : {len(X_test)} ay "
        f"({dates_test.iloc[0].strftime('%Y-%m')} → {dates_test.iloc[-1].strftime('%Y-%m')})"
    )
    logger.info(f"Toplam Feature: {X_train.shape[1]} → Korelasyon ile Top-{TOP_N_FEATURES} seçilecek")

    all_scores = {}
    all_preds  = {}

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Findik_Fiyat_Modelleri")


    # 1. Baseline
    with mlflow.start_run(run_name="Ridge_Baseline"):
        preds_ridge, sc_ridge = train_baseline(X_train, X_test, y_train_log, y_test_raw)
        mlflow.log_metrics({"Test_R2": sc_ridge['R2'], "Test_MAE": sc_ridge['MAE'], 
                            "Test_RMSE": sc_ridge['RMSE'], "Test_MAPE": sc_ridge['MAPE']})
        all_scores['Ridge Baseline'] = sc_ridge
        all_preds['Ridge Baseline']  = preds_ridge

    # 2. XGBoost
    with mlflow.start_run(run_name="XGBoost_Default"):
        xgb_model, xgb_cols, preds_xgb, sc_xgb = train_xgboost(
            X, y_log, X_train, X_test, y_train_log, y_test_raw
        )
        mlflow.log_metrics({"Test_R2": sc_xgb['R2'], "Test_MAE": sc_xgb['MAE'], 
                            "Test_RMSE": sc_xgb['RMSE'], "Test_MAPE": sc_xgb['MAPE']})
        all_scores['XGBoost'] = sc_xgb
        all_preds['XGBoost']  = preds_xgb

    # 3. LightGBM + SHAP
    with mlflow.start_run(run_name="LightGBM_Default"):
        lgb_model, lgb_cols, preds_lgb, sc_lgb = train_lightgbm(
            X, y_log, X_train, X_test, y_train_log, y_test_raw
        )
        mlflow.log_metrics({"Test_R2": sc_lgb['R2'], "Test_MAE": sc_lgb['MAE'], 
                            "Test_RMSE": sc_lgb['RMSE'], "Test_MAPE": sc_lgb['MAPE']})
        all_scores['LightGBM'] = sc_lgb
        all_preds['LightGBM']  = preds_lgb

    # 4. Optuna
    with mlflow.start_run(run_name="Optuna_Best_Models"):
        preds_xgb_o, sc_xgb_o, preds_lgb_o, sc_lgb_o = optuna_optimize(
            X_train, X_test, y_train_log, y_test_raw, xgb_cols, n_trials=50
        )
        # Sadece XGBoost Optuna'nin sonuclarini bu run'a yazalim
        mlflow.log_metrics({"Test_R2_XGB_Optuna": sc_xgb_o['R2'], "Test_MAE_XGB_Optuna": sc_xgb_o['MAE'], 
                            "Test_MAPE_XGB_Optuna": sc_xgb_o['MAPE']})
        
        all_scores['XGBoost (Optuna)']  = sc_xgb_o
        all_scores['LightGBM (Optuna)'] = sc_lgb_o
        all_preds['XGBoost (Optuna)']   = preds_xgb_o
        all_preds['LightGBM (Optuna)']  = preds_lgb_o

    # Tahmin Grafiği

    plot_predictions(y_test_raw, all_preds, dates_test)

    # Özet
    print_summary(all_scores)


if __name__ == "__main__":
    main()

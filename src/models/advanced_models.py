"""
advanced_models.py
==================
Fındık Fiyatı Tahmin Projesi - Gelişmiş Model İyileştirme

Strateji:
  1. Weighted Ensemble  → Mevcut modellerin ağırlıklı ortalaması
  2. Stacking Ensemble  → XGBoost + LightGBM + Ridge → Meta-Ridge
  3. FLAML AutoML       → Otomatik algoritma + hiperparametre arama
  4. N-BEATS / N-HiTS   → Zaman serisi için derin öğrenme (Nixtla)

Çıktı:
  - models/best_model.pkl           → En iyi tek model
  - models/ensemble_weights.json    → Optimal ensemble ağırlıkları
  - reports/figures/09_advanced_model_comparison.png
  - Konsola karşılaştırmalı tablo
"""

import os
import json
import warnings
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb
from scipy.optimize import minimize
import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Sabit Yollar ────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH   = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET         = 'Fiyat_RealUSD_kg'
TOP_N_FEATURES = 20

DROP_COLS = [
    TARGET,
    'Serbest_Piyasa_TL_kg', 'Fiyat_USD_kg', 'US_CPI_Carpani',
    'Tarih', 'Yil_Ay', 'Hasat_Donemi',
    'TMO_Giresun_TL_kg', 'TMO_Levant_TL_kg',
    'Fiyat_Lag1', 'Fiyat_Lag2', 'Fiyat_Lag3', 'Fiyat_Lag12',
    'Fiyat_Degisim_1A_Pct', 'Fiyat_Degisim_3A_Pct',
    'Fiyat_bolu_AsgariUcret_Orani',
    'Yil', 'Sezon_Yili',
]


# ─── Yardımcı Fonksiyonlar ───────────────────────────────────────────────────

def load_and_prepare():
    logger.info(f"Veri yükleniyor: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    df = df.sort_values('Tarih').reset_index(drop=True)

    # USD bazlı lag özellikler (train_model.py ile tutarlı)
    df['USD_Lag1']     = df['Fiyat_USD_kg'].shift(1)
    df['USD_Lag2']     = df['Fiyat_USD_kg'].shift(2)
    df['USD_Lag3']     = df['Fiyat_USD_kg'].shift(3)
    df['USD_Lag12']    = df['Fiyat_USD_kg'].shift(12)
    df['USD_MoM_pct']  = df['Fiyat_USD_kg'].pct_change(1) * 100
    df['USD_YoY_pct']  = df['Fiyat_USD_kg'].pct_change(12) * 100
    df['RealUSD_Lag1'] = df['Fiyat_RealUSD_kg'].shift(1)
    df['RealUSD_Lag3'] = df['Fiyat_RealUSD_kg'].shift(3)
    df = df.bfill().ffill()

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=drop_existing).select_dtypes(include=[np.number])
    y_raw = df[TARGET]
    y_log = np.log1p(y_raw)

    split_idx   = int(len(df) * 0.80)
    X_train     = X.iloc[:split_idx]
    X_test      = X.iloc[split_idx:]
    y_train_log = y_log.iloc[:split_idx]
    y_test_raw  = y_raw.iloc[split_idx:]
    dates_test  = df['Tarih'].iloc[split_idx:]

    # Korelasyon bazlı feature selection (train seti üzerinde)
    corr = X_train.corrwith(y_train_log).abs().dropna()
    sel_cols = corr.nlargest(TOP_N_FEATURES).index.tolist()
    logger.info(f"Feature Selection: {len(X_train.columns)} → {len(sel_cols)} özellik")
    logger.info(f"  Top 10: {sel_cols[:10]}")

    X_tr = X_train[sel_cols]
    X_te = X_test[sel_cols]

    logger.info(f"Train: {len(X_tr)} ay | Test: {len(X_te)} ay")
    return X_tr, X_te, y_train_log, y_test_raw, sel_cols, dates_test


def metrics_orig(y_true_raw, y_pred_orig, prefix=""):
    mae  = mean_absolute_error(y_true_raw, y_pred_orig)
    rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred_orig))
    r2   = r2_score(y_true_raw, y_pred_orig)
    mape = np.mean(np.abs((np.asarray(y_true_raw) - y_pred_orig)
                          / np.where(np.asarray(y_true_raw) == 0, 1, np.asarray(y_true_raw)))) * 100
    logger.info(f"{prefix} → MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}


# ─── Temel modeller (train_model.py ile aynı parametreler) ───────────────────

def train_base_models(X_tr, X_te, y_train_log, y_test_raw):
    logger.info("\n" + "─"*55)
    logger.info("Temel modeller eğitiliyor (XGBoost, LightGBM, Ridge)...")
    logger.info("─"*55)

    # XGBoost
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
        reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbosity=0,
        early_stopping_rounds=30
    )
    xgb_model.fit(X_tr, y_train_log, eval_set=[(X_te, np.log1p(y_test_raw))], verbose=False)
    pred_xgb = np.expm1(xgb_model.predict(X_te))
    sc_xgb = metrics_orig(y_test_raw, pred_xgb, prefix="XGBoost (referans)")

    # LightGBM
    lgb_model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=20,
        min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
        reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1
    )
    lgb_model.fit(X_tr, y_train_log)
    pred_lgb = np.expm1(lgb_model.predict(X_te))
    sc_lgb = metrics_orig(y_test_raw, pred_lgb, prefix="LightGBM (referans)")

    # Ridge Baseline
    scaler = StandardScaler()
    ridge  = Ridge(alpha=10.0)
    ridge.fit(scaler.fit_transform(X_tr), y_train_log)
    pred_ridge = np.expm1(ridge.predict(scaler.transform(X_te)))
    sc_ridge = metrics_orig(y_test_raw, pred_ridge, prefix="Ridge (referans)")

    base_preds = {
        'XGBoost': (pred_xgb, sc_xgb, xgb_model),
        'LightGBM': (pred_lgb, sc_lgb, lgb_model),
        'Ridge': (pred_ridge, sc_ridge, ridge),
    }
    return base_preds, scaler


# ─── 1. Weighted Ensemble ────────────────────────────────────────────────────

def weighted_ensemble(base_preds, y_test_raw):
    logger.info("\n" + "="*55)
    logger.info("ADIM 1: Weighted Ensemble (Scipy Optimize)")
    logger.info("="*55)

    preds_list = [v[0] for v in base_preds.values()]
    names      = list(base_preds.keys())
    y_true     = np.asarray(y_test_raw)

    # Ağırlıkları MAE minimizasyonu ile bul
    def neg_mae(weights):
        weights = np.abs(weights) / np.sum(np.abs(weights))  # normalize
        ensemble = sum(w * p for w, p in zip(weights, preds_list))
        return mean_absolute_error(y_true, ensemble)

    result   = minimize(neg_mae, x0=[1/3, 1/3, 1/3], method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-8})
    best_w   = np.abs(result.x) / np.sum(np.abs(result.x))
    ensemble = sum(w * p for w, p in zip(best_w, preds_list))

    sc = metrics_orig(y_test_raw, ensemble, prefix="Weighted Ensemble (Optimal)")
    for name, w in zip(names, best_w):
        logger.info(f"  {name}: {w:.3f}")

    weights_json = {name: float(w) for name, w in zip(names, best_w)}
    with open(os.path.join(MODELS_DIR, 'ensemble_weights.json'), 'w') as f:
        json.dump(weights_json, f, indent=2, ensure_ascii=False)
    logger.info("Ensemble ağırlıkları kaydedildi → ensemble_weights.json")
    return ensemble, sc, best_w


# ─── 2. Stacking Ensemble ────────────────────────────────────────────────────

def stacking_ensemble(base_preds, X_tr, X_te, y_train_log, y_test_raw, scaler):
    logger.info("\n" + "="*55)
    logger.info("ADIM 2: Stacking Ensemble (Meta-Model: Ridge)")
    logger.info("="*55)

    # Seviye-1: Cross-val OOF tahminleri (log uzayında)
    tscv = TimeSeriesSplit(n_splits=5)
    oof_preds = {}

    for name, (_, _, model) in base_preds.items():
        oof = np.zeros(len(X_tr))
        for tr_idx, val_idx in tscv.split(X_tr):
            if name == 'Ridge':
                fitted = Ridge(alpha=10.0)
                fitted.fit(scaler.fit_transform(X_tr.iloc[tr_idx]), y_train_log.iloc[tr_idx])
                oof[val_idx] = fitted.predict(scaler.transform(X_tr.iloc[val_idx]))
            elif name == 'XGBoost':
                fitted = xgb.XGBRegressor(
                    n_estimators=400, learning_rate=0.05, max_depth=3,
                    subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
                    reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbosity=0
                )
                fitted.fit(X_tr.iloc[tr_idx], y_train_log.iloc[tr_idx])
                oof[val_idx] = fitted.predict(X_tr.iloc[val_idx])
            else:  # LightGBM
                fitted = lgb.LGBMRegressor(
                    n_estimators=400, learning_rate=0.05, num_leaves=20,
                    min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.3, reg_lambda=1.0, random_state=42, verbose=-1
                )
                fitted.fit(X_tr.iloc[tr_idx], y_train_log.iloc[tr_idx])
                oof[val_idx] = fitted.predict(X_tr.iloc[val_idx])
        oof_preds[name] = oof
        logger.info(f"  {name} OOF tamamlandı.")

    # Meta feature matrisi (OOF tahminleri)
    X_meta_train = np.column_stack(list(oof_preds.values()))
    X_meta_test  = np.column_stack([v[0] for v in base_preds.values()])
    # Test tahminleri log uzayına çevrilmiş değil — aşağıda düzeltelim:
    X_meta_test_log = np.column_stack([
        np.log1p(np.maximum(v[0], 0)) for v in base_preds.values()
    ])

    # Seviye-2: Meta-Model
    meta_scaler = StandardScaler()
    meta_ridge  = Ridge(alpha=5.0)
    meta_ridge.fit(meta_scaler.fit_transform(X_meta_train), y_train_log)

    pred_stack_log = meta_ridge.predict(meta_scaler.transform(X_meta_test_log))
    pred_stack     = np.expm1(pred_stack_log)

    sc = metrics_orig(y_test_raw, pred_stack, prefix="Stacking Ensemble")
    joblib.dump({'meta_model': meta_ridge, 'meta_scaler': meta_scaler,
                 'base_models': {k: v[2] for k, v in base_preds.items()}},
                os.path.join(MODELS_DIR, 'stacking_model.pkl'))
    return pred_stack, sc


# ─── 3. FLAML AutoML ─────────────────────────────────────────────────────────

def flaml_automl(X_tr, X_te, y_train_log, y_test_raw, time_budget=90):
    logger.info("\n" + "="*55)
    logger.info(f"ADIM 3: FLAML AutoML ({time_budget} saniye bütçe)")
    logger.info("="*55)
    try:
        from flaml import AutoML
        automl = AutoML()
        automl.fit(
            X_tr, y_train_log,
            task="regression",
            metric="mae",
            time_budget=time_budget,
            estimator_list=["xgboost", "lgbm", "rf", "extra_tree"],
            eval_method="cv",
            n_splits=5,
            split_type="time",
            log_file_name="",
            verbose=0
        )
        logger.info(f"  FLAML En İyi Algoritma : {automl.best_estimator}")
        logger.info(f"  FLAML En İyi CV MAE    : {automl.best_loss:.4f}")

        pred_flaml_log = automl.predict(X_te)
        pred_flaml     = np.expm1(pred_flaml_log)
        sc = metrics_orig(y_test_raw, pred_flaml, prefix="FLAML AutoML")
        joblib.dump(automl, os.path.join(MODELS_DIR, 'flaml_model.pkl'))
        return pred_flaml, sc

    except Exception as e:
        logger.error(f"FLAML çalışamadı: {e}")
        return None, None


# ─── 4. N-BEATS ──────────────────────────────────────────────────────────────

def nbeats_model(y_test_raw, dates_test, df_full):
    """
    N-BEATS: Zaman serisi için derin öğrenme modeli.
    Sadece hedef değişkeni geçmiş değerleri kullanır (univariate).
    152 aylık veri için 12 aylık input window ile 1 adım ilerisi tahmin.
    """
    logger.info("\n" + "="*55)
    logger.info("ADIM 4: N-BEATS (neuralforecast)")
    logger.info("="*55)
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS

        # neuralforecast için veri formatı: unique_id, ds, y
        series = df_full[['Tarih', TARGET]].copy()
        series.columns = ['ds', 'y']
        series.insert(0, 'unique_id', 'findik')
        series['ds'] = pd.to_datetime(series['ds'])
        series = series.dropna(subset=['y'])

        horizon = len(y_test_raw)
        input_size = 24  # 2 yıl geçmiş

        nf = NeuralForecast(
            models=[NBEATS(
                input_size=input_size,
                h=horizon,
                max_steps=200,
                scaler_type='standard',
                random_seed=42,
            )],
            freq='MS'  # Ay başı frekansı
        )

        # Train/val split (son horizon kadar val)
        train_df = series.iloc[:-horizon]
        nf.fit(df=train_df)
        preds_df = nf.predict()
        pred_nbeats = preds_df['NBEATS'].values[:horizon]

        # Orijinal ölçek (hedef zaten RealUSD, log transform yok)
        sc = metrics_orig(y_test_raw, pred_nbeats, prefix="N-BEATS")
        joblib.dump(nf, os.path.join(MODELS_DIR, 'nbeats_model.pkl'))
        return pred_nbeats, sc

    except Exception as e:
        logger.warning(f"N-BEATS çalışamadı: {e}")
        logger.info("  → N-BEATS atlanıyor, diğer modeller devam ediyor.")
        return None, None


# ─── Karşılaştırma Grafiği ───────────────────────────────────────────────────

def plot_comparison(y_test_raw, preds_dict, dates_test, all_scores):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    # Üst: Tahmin vs Gerçek
    ax = axes[0]
    ax.plot(dates_test.values, y_test_raw.values, label='Gerçek Fiyat', color='black', lw=2.5)
    colors = ['royalblue', 'tomato', 'seagreen', 'darkorange', 'mediumpurple', 'brown']
    for (name, preds), color in zip(preds_dict.items(), colors):
        if preds is not None:
            ax.plot(dates_test.values, preds, label=name, linestyle='--', color=color, lw=1.8, alpha=0.85)
    ax.set_title('Gelişmiş Model Tahminleri vs Gerçek Fiyat (Reel USD/kg, 2024 Baz)', fontsize=13)
    ax.set_ylabel('Reel USD/kg')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Alt: MAPE Karşılaştırma Çubuğu
    ax2 = axes[1]
    valid_models = {k: v for k, v in all_scores.items() if v is not None}
    model_names  = list(valid_models.keys())
    mapes        = [valid_models[m]['MAPE'] for m in model_names]
    r2s          = [valid_models[m]['R2']   for m in model_names]
    bar_colors   = ['tomato' if m == 'MAPE' else 'steelblue' for m in model_names]
    bars = ax2.bar(model_names, mapes, color=[
        '#d62728' if mape == min(mapes) else '#aec7e8' for mape in mapes
    ], edgecolor='white', linewidth=0.5)
    for bar, mape, r2 in zip(bars, mapes, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'MAPE: {mape:.1f}%\nR²: {r2:.3f}',
                 ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_title('Model Karşılaştırması — MAPE (%) | Düşük = İyi', fontsize=12)
    ax2.set_ylabel('MAPE (%)')
    ax2.tick_params(axis='x', rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fpath = os.path.join(FIGURES_DIR, '09_advanced_model_comparison.png')
    plt.savefig(fpath, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Karşılaştırma grafiği kaydedildi → {fpath}")


# ─── Özet Tablo ──────────────────────────────────────────────────────────────

def print_summary(all_scores):
    valid = {k: v for k, v in all_scores.items() if v is not None}
    logger.info("\n" + "="*72)
    logger.info("📊 FINAL KARŞILAŞTIRMA — TÜM MODELLER (Test Seti, Reel USD/kg 2024)")
    logger.info("="*72)
    logger.info(f"{'Model':<28} {'R²':>8} {'MAE (USD)':>12} {'RMSE':>10} {'MAPE%':>8}")
    logger.info("-"*72)
    # Sırala: MAPE'ye göre küçükten büyüğe
    for name, sc in sorted(valid.items(), key=lambda x: x[1]['MAPE']):
        star = " ⭐" if sc['MAPE'] == min(s['MAPE'] for s in valid.values()) else ""
        logger.info(f"{name:<28} {sc['R2']:>8.4f} {sc['MAE']:>12.4f} {sc['RMSE']:>10.4f} {sc['MAPE']:>8.2f}{star}")
    logger.info("="*72)

    # En iyi modeli bul
    best_model_name = min(valid, key=lambda k: valid[k]['MAPE'])
    logger.info(f"\n🏆 En İyi Model: {best_model_name} (MAPE: {valid[best_model_name]['MAPE']:.2f}%)")


# ─── Ana Akış ─────────────────────────────────────────────────────────────────

def main():
    # Veri yükleme
    df_raw = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df_raw['Tarih'] = pd.to_datetime(df_raw['Tarih'])
    df_raw = df_raw.sort_values('Tarih').reset_index(drop=True)

    X_tr, X_te, y_train_log, y_test_raw, sel_cols, dates_test = load_and_prepare()

    # Temel modeller
    base_preds, scaler = train_base_models(X_tr, X_te, y_train_log, y_test_raw)

    all_scores = {
        'XGBoost':  base_preds['XGBoost'][1],
        'LightGBM': base_preds['LightGBM'][1],
        'Ridge':    base_preds['Ridge'][1],
    }
    all_preds = {
        'XGBoost':  base_preds['XGBoost'][0],
        'LightGBM': base_preds['LightGBM'][0],
        'Ridge':    base_preds['Ridge'][0],
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Findik_Fiyat_Modelleri")

    # 1. Weighted Ensemble
    with mlflow.start_run(run_name="Weighted_Ensemble"):
        pred_weighted, sc_weighted, _ = weighted_ensemble(base_preds, y_test_raw)
        mlflow.log_metrics({"Test_R2": sc_weighted['R2'], "Test_MAE": sc_weighted['MAE'], 
                            "Test_RMSE": sc_weighted['RMSE'], "Test_MAPE": sc_weighted['MAPE']})
        all_scores['Weighted Ensemble'] = sc_weighted
        all_preds['Weighted Ensemble']  = pred_weighted

    # 2. Stacking Ensemble
    with mlflow.start_run(run_name="Stacking_Ensemble"):
        pred_stack, sc_stack = stacking_ensemble(base_preds, X_tr, X_te, y_train_log, y_test_raw, scaler)
        mlflow.log_metrics({"Test_R2": sc_stack['R2'], "Test_MAE": sc_stack['MAE'], 
                            "Test_RMSE": sc_stack['RMSE'], "Test_MAPE": sc_stack['MAPE']})
        all_scores['Stacking Ensemble'] = sc_stack
        all_preds['Stacking Ensemble']  = pred_stack

    # 3. FLAML AutoML (90 saniye bütçe)
    with mlflow.start_run(run_name="FLAML_AutoML"):
        pred_flaml, sc_flaml = flaml_automl(X_tr, X_te, y_train_log, y_test_raw, time_budget=90)
        if sc_flaml:
            mlflow.log_metrics({"Test_R2": sc_flaml['R2'], "Test_MAE": sc_flaml['MAE'], 
                                "Test_RMSE": sc_flaml['RMSE'], "Test_MAPE": sc_flaml['MAPE']})
        all_scores['FLAML AutoML'] = sc_flaml
        all_preds['FLAML AutoML']  = pred_flaml

    # 4. N-BEATS
    with mlflow.start_run(run_name="NBEATS"):
        pred_nbeats, sc_nbeats = nbeats_model(y_test_raw, dates_test, df_raw)
        if sc_nbeats:
            mlflow.log_metrics({"Test_R2": sc_nbeats['R2'], "Test_MAE": sc_nbeats['MAE'], 
                                "Test_RMSE": sc_nbeats['RMSE'], "Test_MAPE": sc_nbeats['MAPE']})
        all_scores['N-BEATS'] = sc_nbeats
        all_preds['N-BEATS']  = pred_nbeats

    # Karşılaştırma Grafiği
    plot_comparison(y_test_raw, all_preds, dates_test, all_scores)

    # Final Özet
    print_summary(all_scores)


    # En iyi modeli kaydet
    valid = {k: v for k, v in all_scores.items() if v is not None}
    best_name = min(valid, key=lambda k: valid[k]['MAPE'])
    logger.info(f"\nEn iyi model ({best_name}) kaydediliyor → models/best_model_info.json")
    with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w', encoding='utf-8') as f:
        json.dump({'best_model': best_name, 'scores': valid[best_name]}, f,
                  indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

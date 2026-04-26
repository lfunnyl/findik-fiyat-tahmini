import os
import re

file_path = r"c:\Users\funny\Code\lfunnyl\findik-fiyat-tahmini\src\models\train_model.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update prepare_xy
new_prepare_xy = """def prepare_xy(df):
    \"\"\"
    Feature ve Target matrislerini hazırlar.
    Hedef: log(Y_t) - log(Y_t-1) (Delta Log_Return)
    \"\"\"
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
    
    return X_raw, y_log_diff, y_log_prev, y_raw"""

content = re.sub(r'def prepare_xy\(df\):.*?return X_raw, y_log, y_raw', new_prepare_xy, content, flags=re.DOTALL)

# 2. Update metrics_log to just take y_true_orig and y_pred_orig directly
new_metrics_log = """def metrics_log(y_true, y_pred, prefix=""):
    \"\"\"Orijinal ölçekteki tahminleri (USD bazında) alır, metrikleri hesapla.\"\"\"
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    logger.info(f"{prefix} → MAE: {mae:.3f} USD/kg | RMSE: {rmse:.3f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape, 'y_pred_orig': y_pred}"""

content = re.sub(r'def metrics_log\(y_true_log, y_pred_log, prefix=""\):.*?return \{\'MAE\': mae, \'RMSE\': rmse, \'R2\': r2, \'MAPE\': mape, \'y_pred_orig\': y_pred\}', new_metrics_log, content, flags=re.DOTALL)

# 3. Update walk_forward_expanding_cv
new_walk_forward = """def walk_forward_expanding_cv(model_factory, X, y_log_diff, y_log_prev, n_splits=5, model_name="Model"):
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
    return cv_scores"""

content = re.sub(r'def walk_forward_expanding_cv.*?return cv_scores', new_walk_forward, content, flags=re.DOTALL)

# 4. Update train_baseline
new_train_baseline = """def train_baseline(X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\\n" + "="*60)
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
    return scores['y_pred_orig'], scores"""

content = re.sub(r'def train_baseline\(X_train, X_test, y_train_log, y_test_raw\):.*?return scores\[\'y_pred_orig\'\], scores', new_train_baseline, content, flags=re.DOTALL)

# 5. Update train_xgboost
new_train_xgboost = """def train_xgboost(X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\\n" + "="*60)
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
    return xgb_model, sel_cols, scores['y_pred_orig'], scores"""

content = re.sub(r'def train_xgboost\(X, y_log, X_train, X_test, y_train_log, y_test_raw\):.*?return xgb_model, sel_cols, scores\[\'y_pred_orig\'\], scores', new_train_xgboost, content, flags=re.DOTALL)

# 6. Update train_lightgbm
new_train_lightgbm = """def train_lightgbm(X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\\n" + "="*60)
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
    generate_shap_summary(lgb_model, X_te_sel, "LightGBM")
    return lgb_model, sel_cols, scores['y_pred_orig'], scores"""

content = re.sub(r'def train_lightgbm\(X, y_log, X_train, X_test, y_train_log, y_test_raw\):.*?return lgb_model, sel_cols, scores\[\'y_pred_orig\'\], scores', new_train_lightgbm, content, flags=re.DOTALL)

# 7. Update train_catboost
new_train_catboost = """def train_catboost(X, y_log_diff, y_log_prev, X_train, X_test, y_train_log_diff, y_test_log_diff, y_test_log_prev, y_test_raw):
    logger.info("\\n" + "="*60)
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
        return None, None, None, None"""

content = re.sub(r'def train_catboost\(X, y_log, X_train, X_test, y_train_log, y_test_raw\):.*?return None, None, None, None', new_train_catboost, content, flags=re.DOTALL)

# 8. Update optuna_optimize
new_optuna_optimize = """def optuna_optimize(X_train, X_test, y_train_log_diff, y_test_log_prev, y_test_raw, sel_cols, n_trials=50):
    logger.info("\\n" + "="*60)
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

    return sc_xgb['y_pred_orig'], sc_xgb, sc_lgb['y_pred_orig'], sc_lgb"""

content = re.sub(r'def optuna_optimize\(X_train, X_test, y_train_log, y_test_raw, sel_cols, n_trials=50\):.*?return sc_xgb\[\'y_pred_orig\'\], sc_xgb, sc_lgb\[\'y_pred_orig\'\], sc_lgb', new_optuna_optimize, content, flags=re.DOTALL)

# 9. Update main
new_main = """def main():
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
        logger.info(f"\\n[STRESS TEST] Test setinde {shock_mask.sum()} adet Şok Dönemi (>%10 değişim) bulundu.")
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
    logger.info(f"Tüm model skorları kaydedildi → {json_path}")"""

content = re.sub(r'def main\(\):.*?logger.info\(f"Tüm model skorları kaydedildi → \{json_path\}"\)', new_main, content, flags=re.DOTALL)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Replacement complete.")

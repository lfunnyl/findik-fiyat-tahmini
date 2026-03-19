"""
causal_inference.py
===================
Double Machine Learning (DML) Yöntemiyle Nedensel Çıkarım (Causal Inference)

Amaç: Döviz kurunun (USD/TRY) Serbest Piyasa Fındık Fiyatı üzerindeki "saf nedensel" 
etkisini (Causal Effect) hesaplamaktır. Sadece korelasyona bakılmaz; iklim, enflasyon 
ve diğer makro değişkenlerin "Confounder" (yanıltıcı etki) baskısı DML ile izole edilir.

Algoritma:
 1. X_confounders ile T_treatment (Kur) tahmin edilir. Artıklar (t_tilde) bulunur.
 2. X_confounders ile Y_outcome (Fiyat) tahmin edilir. Artıklar (y_tilde) bulunur.
 3. y_tilde ile t_tilde regresyona sokulur. Eğimi ATE (Averate Treatment Effect)'tir.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

def perform_dml():
    logger.info("Double Machine Learning (DML) Nedensellik Analizi baslatiliyor...")
    df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
    df = df.bfill().ffill()
    
    # Hedef (Outcome) ve Mudehale (Treatment) Tanimlari
    Y_col = 'Serbest_Piyasa_TL_kg'
    T_col = 'USD_TRY_Kapanis'
    
    if Y_col not in df.columns or T_col not in df.columns:
        logger.error("Gerekli kolonlar bulunamadi!")
        return
        
    Y = df[Y_col].values
    T = df[T_col].values
    
    # Confounders (Diger her sey - Makro, iklim vs.)
    drop_confhounds = [
        Y_col, T_col, 'Tarih', 'Yil_Ay', 'Fiyat_USD_kg', 'Fiyat_RealUSD_kg', 
        'TMO_Giresun_TL_kg', 'TMO_Levant_TL_kg'
    ]
    X_df = df.drop(columns=[c for c in drop_confhounds if c in df.columns]).select_dtypes(include=[np.number])
    X = X_df.values
    
    logger.info(f"Confounder Sayisi: {X.shape[1]}")
    
    # Model secimi
    model_t = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model_y = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    
    # Adim 1 & 2: Cross Validation ile residuals hesapla (Overfitting'i onlemek icin)
    # T_hat = M(X), Y_hat = M(X)
    t_hat = cross_val_predict(model_t, X, T, cv=5)
    y_hat = cross_val_predict(model_y, X, Y, cv=5)
    
    t_tilde = T - t_hat
    y_tilde = Y - y_hat
    
    # Adim 3: Final Nedensel Etki (ATE) Regresyonu
    # y_tilde = theta * t_tilde + epsilon
    # theta = sum(t_tilde * y_tilde) / sum(t_tilde^2)
    theta = np.sum(t_tilde * y_tilde) / np.sum(t_tilde**2)
    
    logger.info("="*50)
    logger.info(f"DML Sonucu: Ortalama Tedavi Etkisi (ATE) = {theta:.3f}")
    logger.info(f"Yorum: Kurdaki 1 birimlik (1.00 TL) nedensel artis, "
                f"enflasyon ve makro statik varsayilirsa fındık fiyatını ortalama "
                f"{theta:.2f} TL artiriyor.")
    logger.info("="*50)
    
    # Gorsellestirme
    plt.figure(figsize=(8, 6))
    plt.scatter(t_tilde, y_tilde, alpha=0.5, color='#7c6af7')
    
    # Trend line
    x_vals = np.linspace(min(t_tilde), max(t_tilde), 100)
    y_vals = theta * x_vals
    plt.plot(x_vals, y_vals, color='#ff5722', linewidth=2, label=f'Causal Slope (ATE): {theta:.2f}')
    
    plt.title('Double Machine Learning: Kurun Fiyata Saf Etkisi (Causal Interface)', fontsize=12)
    plt.xlabel('Residual Treatment (Kur - Beklenen Kur)', fontsize=10)
    plt.ylabel('Residual Outcome (Fiyat - Beklenen Fiyat)', fontsize=10)
    plt.grid(alpha=0.3)
    plt.legend()
    
    plot_path = os.path.join(REPORTS_DIR, '12_causal_usd_effect.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # JSON Rapor Olustur
    res_json = {
        "treatment": T_col,
        "outcome": Y_col,
        "average_treatment_effect": round(float(theta), 3),
        "yorum": f"Kurdaki 1 birimlik sapsaf artış, izole nedensellikle fiyatı ortalama {round(float(theta), 2)} TL yukarı doğru çekmektedir."
    }
    json_path = os.path.join(BASE_DIR, "models", "causal_effect.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(res_json, f, ensure_ascii=False, indent=4)
        
    logger.info(f"Raporlar kaydedildi: {plot_path}, {json_path}")


if __name__ == "__main__":
    perform_dml()

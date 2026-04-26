import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitim ve Tahmin (API) için ortak özellik mühendisliği adımları.
    Tüm modelleme (Delta Modeling, Regime Detection, Causal Forcing) 
    için gerekli türetilmiş özellikleri merkezileştirir.
    """
    df = df.copy()
    
    # Tarih sıralaması kritik
    if 'Tarih' in df.columns:
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df = df.sort_values('Tarih').reset_index(drop=True)

    # 1. Delta (Değişim) Özellikleri - shift(1) ile sızıntı önlenir
    if 'Fiyat_USD_kg' in df.columns:
        df["USD_MoM_pct"]     = df["Fiyat_USD_kg"].shift(1).pct_change(1) * 100
        df["USD_YoY_pct"]     = df["Fiyat_USD_kg"].shift(1).pct_change(12) * 100
        
    if 'Fiyat_RealUSD_kg' in df.columns:
        df["RealUSD_MoM_pct"] = df["Fiyat_RealUSD_kg"].shift(1).pct_change(1) * 100
        df["RealUSD_YoY_pct"] = df["Fiyat_RealUSD_kg"].shift(1).pct_change(12) * 100
        
        # Hareketli Ortalamalar
        df['RealUSD_MA3'] = df['Fiyat_RealUSD_kg'].shift(1).rolling(window=3).mean()
        df['RealUSD_MA6'] = df['Fiyat_RealUSD_kg'].shift(1).rolling(window=6).mean()
        df['Fiyat_MA3_Farki_Pct'] = (df['Fiyat_RealUSD_kg'].shift(1) - df['RealUSD_MA3']) / df['RealUSD_MA3'] * 100

    # 2. Döviz ve Volatilite Özellikleri
    if 'USD_TRY_Kapanis' in df.columns:
        df['Kur_Aylik_Ivme']  = df['USD_TRY_Kapanis'].shift(1).pct_change(1) * 100
        df['Kur_Volatilite_3Ay'] = df['USD_TRY_Kapanis'].shift(1).rolling(window=3).std()

    # Eksik verileri doldur
    df = df.bfill().ffill()

    # 3. Regime Detection (Şok Alarmı)
    if 'Kur_Volatilite_3Ay' in df.columns:
        volatilite_mean = df['Kur_Volatilite_3Ay'].mean()
        is_shock = (df['Kur_Volatilite_3Ay'] > volatilite_mean * 2)
        if 'Kritik_Don' in df.columns:
            is_shock = is_shock | (df['Kritik_Don'] > 0)
        df['Regime_Shock_Warning'] = np.where(is_shock, 1, 0)

    # 4. TMO Müdahalesi (Policy Causal Feature)
    if 'TMO_Giresun_TL_kg' in df.columns and 'Serbest_Piyasa_TL_kg' in df.columns:
        df['TMO_Fiyat_Artis_Pct'] = df['TMO_Giresun_TL_kg'].pct_change(1) * 100
        df['TMO_Mevcut_Makas_Pct'] = (df['TMO_Giresun_TL_kg'] - df['Serbest_Piyasa_TL_kg'].shift(1)) / df['Serbest_Piyasa_TL_kg'].shift(1) * 100

    return df

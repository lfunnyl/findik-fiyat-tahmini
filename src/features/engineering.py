import pandas as pd
import numpy as np

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eğitim ve Tahmin (API) için ortak özellik mühendisliği adımları.
    Girdi: Ham veya yarı-işlenmiş DataFrame
    Çıktı: Lag ve oran özelliklerini içeren zenginleştirilmiş DataFrame
    """
    df = df.copy()
    
    # Tarih sıralaması kritik
    if 'Tarih' in df.columns:
        df['Tarih'] = pd.to_datetime(df['Tarih'])
        df = df.sort_values('Tarih').reset_index(drop=True)

    # USD Lag Özellikleri
    if 'Fiyat_USD_kg' in df.columns:
        df["USD_Lag1"]     = df["Fiyat_USD_kg"].shift(1)
        df["USD_Lag2"]     = df["Fiyat_USD_kg"].shift(2)
        df["USD_Lag3"]     = df["Fiyat_USD_kg"].shift(3)
        df["USD_Lag12"]    = df["Fiyat_USD_kg"].shift(12)
        df["USD_MoM_pct"]  = df["Fiyat_USD_kg"].pct_change(1) * 100
        df["USD_YoY_pct"]  = df["Fiyat_USD_kg"].pct_change(12) * 100
        
    # Real USD Lag Özellikleri
    if 'Fiyat_RealUSD_kg' in df.columns:
        df["RealUSD_Lag1"] = df["Fiyat_RealUSD_kg"].shift(1)
        df["RealUSD_Lag3"] = df["Fiyat_RealUSD_kg"].shift(3)
        df["RealUSD_Lag12"] = df["Fiyat_RealUSD_kg"].shift(12)

    # Eksik verileri doldur
    df = df.bfill().ffill()
    
    return df

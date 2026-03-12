"""
build_features.py
=================
Bu script data/processed/ klasöründeki tüm "_temiz.csv" dosyalarını alır,
aylık bazda (granülarite) birleştirir ve makine öğrenmesi modeli için
gerekli olan anlamlı öznitelikleri (features) üretir.

Oluşturulan Çıktı: data/processed/master_features.csv
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")

def load_clean_data(filename):
    filepath = os.path.join(PROC_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"Dosya bulunamadı: {filepath}")
        return None
    return pd.read_csv(filepath, encoding='utf-8-sig')

def process_daily_to_monthly(df, date_col, aggregations):
    """Günlük veriyi aylık (YYYY-MM) bazında gruplar."""
    if df is None or df.empty:
        return None
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['Yil_Ay'] = df[date_col].dt.to_period('M')
    
    # Aggregation
    df_monthly = df.groupby('Yil_Ay').agg(aggregations).reset_index()
    # Sütun isimlerini düzleştir
    if isinstance(df_monthly.columns, pd.MultiIndex):
        df_monthly.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_monthly.columns.values]
    return df_monthly

def build_features():
    logger.info("Feature Engineering başlatılıyor...")

    # 1. Ana hedef tablomuz (Aylık Fındık Fiyatları)
    df_fiyat = load_clean_data("turkiye_findik_fiyatlari_temiz.csv")
    df_fiyat['Tarih'] = pd.to_datetime(df_fiyat['Tarih'])
    df_fiyat['Yil_Ay'] = df_fiyat['Tarih'].dt.to_period('M')
    df_fiyat['Yil'] = df_fiyat['Tarih'].dt.year
    df_fiyat['Ay'] = df_fiyat['Tarih'].dt.month

    # 2. Günlük Verilerin Aylığa Çevrilmesi
    # Kur verisi ve Kur Volatilitesi hesaplaması
    df_kur = load_clean_data("historical_usd_try_5_years_temiz.csv")
    if df_kur is not None:
        df_kur['Tarih'] = pd.to_datetime(df_kur['Tarih'])
        df_kur = df_kur.sort_values('Tarih')
        df_kur['Kur_Volatilite_30G'] = df_kur['USD_TRY_Kapanis'].rolling(window=30).std()
        df_kur_aylik = process_daily_to_monthly(df_kur, 'Tarih', {
            'USD_TRY_Kapanis': 'mean', 
            'Kur_Volatilite_30G': 'mean'
        })
    else:
        df_kur_aylik = None
    
    # Makro veriler
    df_makro = load_clean_data("makro_veriler_5_years_temiz.csv")
    df_makro_aylik = process_daily_to_monthly(df_makro, 'Tarih', {
        'Brent_Petrol_Kapanis': 'mean',
        'Altin_Ons_Kapanis': 'mean',
        'Dogalgaz_ETF_Kapanis': 'mean',
        'Kakao_Kapanis': 'mean'
    })

    # Asgari Ücret
    df_wage = load_clean_data("turkiye_asgeri_ucret_veri_temiz.csv")
    df_wage_aylik = process_daily_to_monthly(df_wage, 'Tarih', {'Asgari_Ucret_TL': 'mean'})

    # İklim Verisi (Toplama türüne dikkat et)
    df_iklim = load_clean_data("karadeniz_iklim_5_years_temiz.csv")
    df_iklim_aylik = process_daily_to_monthly(df_iklim, 'Tarih', {
        'Max_Sicaklik_C': 'mean',
        'Min_Sicaklik_C': 'min',     # O ayki en düşük sıcaklık don riski için kritik
        'Yagis_mm': 'sum',           # O ayki toplam yağış
        'Kritik_Don': 'sum'          # O aydaki toplam don günü sayısı
    })

    # 3. Yıllık Verilerin Yüklenmesi
    df_tufe = load_clean_data("turkiye_tufe_temiz.csv")             # join on Yil
    df_ihracat = load_clean_data("turkiye_findik_ihracat_temiz.csv") # join on Yil
    df_fao = load_clean_data("fao_findik_uretim_temiz.csv")         # join on Yil
    df_rekolte = load_clean_data("rekolte_arz_talep_temiz.csv")     # join on Sezon_Yili
    df_ekstra = load_clean_data("ekstra_makro_Sosyolojik_temiz.csv") # join on Yil_Ay

    # ── MERGE İŞLEMLERİ ──
    logger.info("Veri setleri birleştiriliyor...")
    df_master = df_fiyat.copy()

    # Günlükten aylığa dönenleri 'Yil_Ay' ile birleştir
    if df_kur_aylik is not None: df_master = pd.merge(df_master, df_kur_aylik, on='Yil_Ay', how='left')
    df_master = pd.merge(df_master, df_makro_aylik, on='Yil_Ay', how='left')
    df_master = pd.merge(df_master, df_wage_aylik, on='Yil_Ay', how='left')
    df_master = pd.merge(df_master, df_iklim_aylik, on='Yil_Ay', how='left')
    
    # Ekstra Sosyolojik & Makro Özellikler
    if df_ekstra is not None:
        df_ekstra['Yil_Ay'] = pd.to_datetime(df_ekstra['Yil_Ay']).dt.to_period('M')
        df_master = pd.merge(df_master, df_ekstra, on='Yil_Ay', how='left')

    # Yıllık verileri birleştir (Her aya o yılın verisini koyar - Forward fill benzeri)
    # Ekonomi ve üretim verileri Sezon_Yili değil, Takvim Yılına ('Yil') göre gelir
    df_master = pd.merge(df_master, df_tufe, on='Yil', how='left')
    df_master = pd.merge(df_master, df_ihracat, on='Yil', how='left')
    df_master = pd.merge(df_master, df_fao, on='Yil', how='left')

    # Rekolte verileri 'Sezon_Yili' bazında birleştirilir
    df_master = pd.merge(df_master, df_rekolte, on='Sezon_Yili', how='left')

    # ── FEATURE ENGINEERING (Özellik Türetimi) ──
    logger.info("Yeni özellikler türetiliyor (Feature Engineering)...")

    # 1. USD Bazlı Fiyat (Dolar bazında fındık fiyatı nasıl hareket etti?)
    df_master['Fiyat_USD_kg'] = df_master['Serbest_Piyasa_TL_kg'] / df_master['USD_TRY_Kapanis']

    # 2. Döviz ve Makro Momentum Özellikleri
    df_master['Kur_Aylik_Degisim_Pct'] = df_master['USD_TRY_Kapanis'].pct_change() * 100
    df_master['Altin_Aylik_Degisim_Pct'] = df_master['Altin_Ons_Kapanis'].pct_change() * 100

    # 3. Fındık Fiyatı Momentum / Gecikmeli Özellikler (Lags)
    # Önceki aylardaki fiyatlar tahmin için çok güçlü sinyallerdir.
    df_master['Fiyat_Lag1'] = df_master['Serbest_Piyasa_TL_kg'].shift(1)
    df_master['Fiyat_Lag2'] = df_master['Serbest_Piyasa_TL_kg'].shift(2)
    df_master['Fiyat_Lag3'] = df_master['Serbest_Piyasa_TL_kg'].shift(3)
    df_master['Fiyat_Lag12']= df_master['Serbest_Piyasa_TL_kg'].shift(12)  # Geçen yılın aynı ayı

    df_master['Fiyat_Degisim_1A_Pct'] = df_master['Serbest_Piyasa_TL_kg'].pct_change(1) * 100
    df_master['Fiyat_Degisim_3A_Pct'] = df_master['Serbest_Piyasa_TL_kg'].pct_change(3) * 100

    # 4. Asgari Ücret ve Maliyet Baskısı
    # İşçi maliyetinin fiyata etkisini oransal görmek için
    df_master['Fiyat_bolu_AsgariUcret_Orani'] = (df_master['Serbest_Piyasa_TL_kg'] * 1000) / df_master['Asgari_Ucret_TL']

    # 5. Mevsimsellik (Cyclical Features)
    # Aylar daireseldir, Aralık(12) ile Ocak(1) aslında birbirine yakındır.
    df_master['Ay_Sin'] = np.sin(2 * np.pi * df_master['Ay'] / 12.0)
    df_master['Ay_Cos'] = np.cos(2 * np.pi * df_master['Ay'] / 12.0)

    # 6. NaN Yönetimi
    # Shift işlemi yaptık, ilk 12 satırda Lag12 NaN olacak. Bu eksikleri bfill (geriye dönük) dolduralım
    df_master.bfill(inplace=True)
    df_master.ffill(inplace=True) # İklim verisi vb. API limitinden ötürü baştaki eksiklikler için

    # Kategorik metin sütunlarını modele girmeyecek şekilde ayıkla veya One-Hot encode yap
    durum_mapping = {'Yuksek': 1, 'Normal': 0, 'Dusuk': -1}
    if 'Randiman_Kalite' in df_master.columns:
        df_master['Randiman_Skoru'] = df_master['Randiman_Kalite'].map(durum_mapping).fillna(0)
        df_master.drop(columns=['Randiman_Kalite'], inplace=True, errors='ignore')

    # Yil_Ay kolonunu string yap veya düşür (Tarih verisi asıl Date işlevi görecek)
    df_master['Yil_Ay'] = df_master['Yil_Ay'].astype(str)

    # Kaydetme
    out_file = os.path.join(PROC_DIR, "master_features.csv")
    df_master.to_csv(out_file, index=False, encoding='utf-8-sig')
    
    logger.info(f"Master veri çerçevesi başarıyla oluşturuldu: {out_file}")
    logger.info(f"Oluşturulan Matris Boyutu: {df_master.shape}")
    logger.info(f"\nSütunların Listesi:")
    for i, col in enumerate(df_master.columns):
        if i % 4 == 0:
            print("\n  ", end="")
        print(f"{col[:25]:<27}", end="")
    print("\n")


if __name__ == "__main__":
    build_features()

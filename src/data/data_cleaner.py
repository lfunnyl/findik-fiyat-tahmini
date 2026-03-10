"""
Veri Temizleme ve Kalite Analizi
=================================
Tüm raw CSV dosyalarını tarar ve şunları raporlar:
  1. Eksik değer (NaN) analizi
  2. Tarih format kontrolü
  3. Outlier tespiti (IQR yöntemi)
  4. Veri yeterliliği (train/test split senaryosu)
  5. Temizlenmiş verileri data/processed/ klasörüne yazar
"""

import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

# ── Hangi sütun tarih, hangileri sayısal ──────────────────────────────────────
DOSYA_META = {
    "turkiye_findik_fiyatlari.csv": {
        "tarih_kol": "Tarih",
        "granul": "ay",
        "sayisal": ["TMO_Giresun_TL_kg", "TMO_Levant_TL_kg", "Serbest_Piyasa_TL_kg"],
        "hedef": True,
    },
    "karadeniz_iklim_5_years.csv": {
        "tarih_kol": "Tarih",
        "granul": "gun",
        "sayisal": ["Max_Sicaklik_C", "Min_Sicaklik_C", "Yagis_mm"],
    },
    "makro_veriler_5_years.csv": {
        "tarih_kol": "Tarih",
        "granul": "gun",
        "sayisal": ["Brent_Petrol_Kapanis", "Altin_Ons_Kapanis", "Dogalgaz_ETF_Kapanis"],
    },
    "historical_usd_try_5_years.csv": {
        "tarih_kol": "Tarih",
        "granul": "gun",
        "sayisal": ["USD_TRY_Kapanis"],
    },
    "turkiye_asgeri_ucret_veri.csv": {    # Dosya adı farklı olabilir
        "tarih_kol": "Tarih",
        "granul": "gun",
        "sayisal": ["Asgari_Ucret_TL"],
    },
    "rekolte_arz_talep.csv": {
        "tarih_kol": None,  # Yıl bazlı
        "granul": "yil",
        "sayisal": ["TR_Rekolte_Gercek_Ton", "Rekolte_Surprise_Pct", "Randiman_Pct",
                    "Global_Arz_Acigi_Ton", "Stok_Kullanim_Oran_Pct"],
    },
    "turkiye_tufe.csv": {
        "tarih_kol": None,
        "granul": "yil",
        "sayisal": ["TUFE_Yillik_Pct"],
    },
    "turkiye_findik_ihracat.csv": {
        "tarih_kol": None,
        "granul": "yil",
        "sayisal": ["Ihracat_Ton_Ic_Findik", "Ihracat_Deger_mUSD"],
    },
    "fao_findik_uretim.csv": {
        "tarih_kol": None,
        "granul": "yil",
        "sayisal": ["Uretim_Ton_Turkiye", "Uretim_Ton_Dunyaa"],
    },
}

def dosya_bul(dosya_adi):
    """Dosya adını raw klasöründe case-insensitive arar."""
    for f in os.listdir(RAW_DIR):
        if f.lower().replace("ü","u").replace("ı","i") == dosya_adi.lower().replace("ü","u").replace("ı","i"):
            return os.path.join(RAW_DIR, f)
    return os.path.join(RAW_DIR, dosya_adi)

def outlier_tespit(df, kolonlar, iqr_carpan=3.0):
    """IQR yöntemi ile aykırı değerleri tespit eder."""
    bulgular = {}
    for kol in kolonlar:
        if kol not in df.columns:
            continue
        seri = df[kol].dropna()
        if len(seri) < 4:
            continue
        Q1, Q3 = seri.quantile(0.25), seri.quantile(0.75)
        IQR = Q3 - Q1
        alt_sinir = Q1 - iqr_carpan * IQR
        ust_sinir = Q3 + iqr_carpan * IQR
        aykirilar = seri[(seri < alt_sinir) | (seri > ust_sinir)]
        if len(aykirilar) > 0:
            bulgular[kol] = {
                "adet": len(aykirilar),
                "degerler": aykirilar.values.tolist()[:5],
                "sinirlar": (round(alt_sinir, 2), round(ust_sinir, 2)),
            }
    return bulgular

def tek_dosya_analiz(dosya_adi, meta):
    """Tek bir CSV dosyasını analiz eder."""
    yol = dosya_bul(dosya_adi)
    if not os.path.exists(yol):
        logger.warning(f"  [UYARI]  Dosya bulunamadı: {dosya_adi}")
        return None

    df = pd.read_csv(yol, encoding='utf-8-sig')
    sonuc = {
        "dosya": dosya_adi,
        "satirlar": len(df),
        "kolonlar": len(df.columns),
        "granularite": meta["granul"],
        "nan_raporu": {},
        "outlier_raporu": {},
        "uyarilar": [],
    }

    # ── 1. NaN analizi ──
    for kol in df.columns:
        nan_sayi = df[kol].isna().sum()
        if nan_sayi > 0:
            sonuc["nan_raporu"][kol] = {
                "adet": int(nan_sayi),
                "oran_pct": round(nan_sayi / len(df) * 100, 1),
            }

    # ── 2. Tarih formatı kontrolü ──
    tarih_kol = meta.get("tarih_kol")
    if tarih_kol and tarih_kol in df.columns:
        try:
            df[tarih_kol] = pd.to_datetime(df[tarih_kol])
        except Exception:
            sonuc["uyarilar"].append(f"Tarih parse hatası: {tarih_kol}")

    # ── 3. Sayısal kolonları float'a zorla ──
    for kol in meta.get("sayisal", []):
        if kol in df.columns:
            df[kol] = pd.to_numeric(df[kol], errors='coerce')

    # ── 4. Outlier tespiti ──
    sonuc["outlier_raporu"] = outlier_tespit(df, meta.get("sayisal", []))

    # ── 5. Temizlenmiş versiyonu kaydet ──
    temiz_yol = os.path.join(PROC_DIR, dosya_adi.replace(".csv", "_temiz.csv"))
    df.to_csv(temiz_yol, index=False, encoding='utf-8-sig')

    return sonuc, df

def veri_yeterliligi_analizi():
    """Hedef değişken üzerinden train/test yeterliliği raporlar."""
    print("\n" + "="*70)
    print("[ANALIZ] VERİ YETERLİLİĞİ ANALİZİ")
    print("="*70)

    fiyat_yol = dosya_bul("turkiye_findik_fiyatlari.csv")
    if os.path.exists(fiyat_yol):
        df_hedef = pd.read_csv(fiyat_yol, encoding='utf-8-sig')
        total = len(df_hedef)
        gerc_gozlem = df_hedef["Serbest_Piyasa_TL_kg"].notna().sum()

        print(f"\n[*] Hedef Değişken (aylık fındık fiyatı):")
        print(f"   Toplam aylık satır       : {total}")

        print(f"\n[*] Train/Test Senaryoları:")
        for oran in [0.7, 0.8]:
            train_n = int(total * oran)
            test_n  = total - train_n
            print(f"   %{int(oran*100)} Train / %{100-int(oran*100)} Test  → Train: {train_n} ay | Test: {test_n} ay")

        print(f"\n[UYARI]  DEĞERLENDİRME:")
        if total < 60:
            print(f"   [KRT] KRİTİK: {total} aylık veri çok az! Minimum 60 önerilir.")
        elif total < 100:
            print(f"   [UYR] SINIRDA: {total} aylık veri. XGBoost çalışır, LSTM için yetersiz.")
            print(f"      → Walk-Forward (Kayan Pencere) CV kullan, basit 80/20 değil!")
            print(f"      → Veriyi 2015'e kadar geriye genişlet (ek ~44 ay)")
        else:
            print(f"   [IYI] YETERLİ: {total} aylık veri. XGBoost ve LSTM için uygun.")

        print(f"\n[*] Önerilen Strateji:")
        print(f"   TimeSeriesSplit(n_splits=5)  ← scikit-learn kayan pencere CV")
        print(f"   Her fold: ~{total//6} ay train → bir sonraki ~{total//6} ay test")
        print(f"   Bu yöntem küçük veri setlerinde çok daha güvenilir sonuç verir.")

def rapor_yazdir(sonuclar):
    """Konsola formatlanmış rapor yazar."""
    print("\n" + "="*70)
    print("🔍 VERİ KALİTE RAPORU")
    print("="*70)

    for s in sonuclar:
        if s is None:
            continue
        dosya, sonuc = s[0], s[0]
        print(f"\n[DOS] {sonuc['dosya']}  ({sonuc['satirlar']} satır, {sonuc['granularite']})")

        if sonuc["nan_raporu"]:
            print(f"   [HAT] NaN değerler:")
            for kol, info in sonuc["nan_raporu"].items():
                print(f"      {kol}: {info['adet']} adet ({info['oran_pct']}%)")
        else:
            print(f"   [OK] NaN yok")

        if sonuc["outlier_raporu"]:
            print(f"   [UYARI]  Aykırı değerler (IQR×3):")
            for kol, info in sonuc["outlier_raporu"].items():
                print(f"      {kol}: {info['adet']} adet | sınırlar: {info['sinirlar']}")
        else:
            print(f"   [OK] Belirgin aykırı değer yok")

        if sonuc["uyarilar"]:
            for u in sonuc["uyarilar"]:
                print(f"   [KRT] {u}")


if __name__ == "__main__":
    logger.info("Veri Temizleme ve Kalite Analizi Başlatıldı...")
    print("\n" + "="*70)
    print("[TARAMA] TÜM HAM VERİLER TARANADI")
    print("="*70)

    sonuclar = []
    for dosya_adi, meta in DOSYA_META.items():
        result = tek_dosya_analiz(dosya_adi, meta)
        if result:
            sonuclar.append(result)

    rapor_yazdir(sonuclar)
    veri_yeterliligi_analizi()

    print(f"\n[OK] Temizlenmiş dosyalar → data/processed/ klasörüne kaydedildi.")
    logger.info("Analiz tamamlandı.")

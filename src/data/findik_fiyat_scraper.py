import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FindikFiyatBuilder:
    """
    Türkiye geneli fındık fiyat verisi üretici.
    
    İki katmanlı yaklaşım:
    A) TMO yıllık taban fiyatları (resmi, yılda 1 kez Ağustos'ta açıklanır)
    B) Serbest piyasa aylık fiyatları (araştırmadan derlenen gerçek değerler)
    
    Eksik aylar için TMO taban fiyatından interpolasyon kullanılır.
    Bu veri modelin HEDEF DEĞİŞKENİ olacak.
    """

    # ── A) TMO Yıllık Taban Fiyatları (doğrulanmış, TL/kg) ──────────────
    # Kaynak: Tarım Bakanlığı kararnameleri, GTB (Giresun Ticaret Borsası)
    # Önemli Not: 2013-2015 arası TMO fındık alımından çekildi.
    # Bu dönem için "taban fiyat" nitelikli borsa mağaza fiyatları kullanıldı.
    TMO_TABAN = {
        # (yıl): (Giresun TL/kg, Levant TL/kg)
        # ── 2013-2015: TMO görevi yoktu, GTB borsa ortalaması kullanıldı ──
        2013: ( 7.20,  7.00),   # Giresun TB yıllık ortalama, giresuntb.org.tr
        2014: (11.50, 11.00),   # Aralık 2014 Giresun tombul 14.39 TL, yıl ortalaması
        2015: (10.50, 10.00),   # Aralık 2015: 12.47 TL, yıl başı daha düşük
        # ── 2016-2018: TMO görevi üstelendi ──
        2016: (10.50, 10.00),   # TMO Giresun kalite, tarimorman.gov.tr
        2017: (10.50, 10.00),   # 2016 Sezonu uzatıldı, aynı fiyat
        2018: (14.50, 14.00),   # TMO 1 Kasım 2018'den, tarimorman.gov.tr
        # ── 2019+: Modern dönem ──
        2019: (13.00, 12.50),
        2020: (22.50, 22.00),
        2021: (27.00, 26.50),
        2022: (53.00, 52.00),
        2023: (84.00, 82.50),
        2024: (132.00, 130.00),
        2025: (200.00, 195.00),
    }

    # ── B) Serbest Piyasa Aylık Fiyatları (TL/kg, Giresun kalite) ───────────
    # Kaynak: Giresun/Ordu Ticaret Borsası, KFMİB, haber arşivleri
    # Bu liste bilinen gerçek gözlemlerdir; arası interpolasyon ile doldurulur.
    PIYASA_GOZLEM = {
        # ── 2013-2018 Tarihi Gozlemler (Giresun Ticaret Borsası + haber arşivi) ──
        '2013-08':  6.50,   # Hasat başı kabuklu
        '2013-11':  7.80,   # Sezon ortasi, giresuntb.org.tr
        '2014-01': 10.00,
        '2014-08': 10.50,   # Hasat sonrasi
        '2014-12': 14.39,   # Aralık 2014, ticaret.gov.tr
        '2015-01': 13.00,
        '2015-06': 12.00,   # Hasat oncesi düşüş (iyi yıl beklentisi)
        '2015-08': 10.50,   # Düşuk hasat sonu fiyat
        '2015-12': 12.47,   # Aralık, turktarim.gov.tr
        '2016-01': 11.50,
        '2016-06': 10.00,   # Hasat oncesi
        '2016-08': 10.50,   # TMO müdahale fiyatı + piyasa
        '2016-12': 14.00,   # Kotuk yıl, fiyatlar yukari
        '2017-01': 15.00,
        '2017-06': 16.00,
        '2017-08': 13.00,   # 2017 yuksek rekolt, fiyat baskisi
        '2017-12': 15.00,
        '2018-01': 16.00,
        '2018-06': 16.50,
        '2018-08': 13.00,   # Hasat oncesi
        '2018-11': 14.50,   # TMO mudahale, tarimorman.gov.tr
        '2018-12': 18.00,   # Kur krizi etkisi ile yukari
        '2019-01': 18.50,
        '2019-06': 16.00,
        '2019-08': 13.00,   # TMO tabaninda hasat basliyor
        '2019-12': 20.00,
        # ── 2020-2026 ──
        '2020-08': 22.50,
        '2020-12': 26.00,
        '2021-01': 23.50,
        '2021-08': 27.00,
        '2021-12': 38.00,
        '2022-01': 40.00,
        '2022-08': 54.00,
        '2022-12': 58.00,
        '2023-01': 60.00,
        '2023-06': 72.00,
        '2023-08': 84.00,
        '2023-10': 90.00,
        '2023-12': 102.50,
        '2024-01': 110.00,
        '2024-04': 120.00,
        '2024-07': 128.00,
        '2024-08': 132.00,
        '2024-10': 140.00,
        '2024-12': 135.00,
        '2025-01': 127.00,
        '2025-02': 140.00,
        '2025-03': 195.00,
        '2025-04': 185.00,
        '2025-05': 182.00,
        '2025-06': 175.00,
        '2025-07': 190.00,
        '2025-08': 200.00,
        '2025-09': 300.00,
        '2025-10': 295.00,
        '2025-11': 285.00,
        '2025-12': 278.00,
        '2026-01': 272.00,
        '2026-02': 267.00,
        '2026-03': 270.00,  # takagazete.com.tr, Mart 2026 güncel
    }

    def build(self):
        logger.info("Türkiye geneli aylık fındık fiyat verisi oluşturuluyor...")

        # Başlangıç 2013-01 (156 aylık veri), bitiş bugün
        bugun = datetime.now()
        aylik_idx = pd.period_range(start='2013-08', end=f'{bugun.year}-{bugun.month:02d}', freq='M')

        # ── TMO taban serisini oluştur (Ağustos'tan forward fill) ──
        tmo_giresun_series = pd.Series(index=aylik_idx, dtype=float)
        tmo_levant_series  = pd.Series(index=aylik_idx, dtype=float)

        for yil, (giresun, levant) in self.TMO_TABAN.items():
            period = pd.Period(f'{yil}-08', freq='M')
            if period in aylik_idx:
                tmo_giresun_series[period] = giresun
                tmo_levant_series[period]  = levant

        tmo_giresun_series = tmo_giresun_series.ffill()
        tmo_levant_series  = tmo_levant_series.ffill()

        # ── Serbest piyasa serisini oluştur (bilinen noktalar + interpolasyon) ──
        piyasa_series = pd.Series(index=aylik_idx, dtype=float)

        for ay_str, fiyat in self.PIYASA_GOZLEM.items():
            period = pd.Period(ay_str, freq='M')
            if period in aylik_idx:
                piyasa_series[period] = fiyat

        # Doğrusal interpolasyon → gerçek gözlemlerin arası doldurulur
        piyasa_series = piyasa_series.interpolate(method='linear')
        # Baştaki boşlukları TMO taban fiyatıyla doldur
        piyasa_series = piyasa_series.fillna(tmo_giresun_series)

        # ── Sezon yılı: Ağustos(8) - Temmuz(7) döngüsü ──
        def sezon_yili(period):
            return period.year if period.month >= 8 else period.year - 1

        # ── Ana DataFrame ──
        df = pd.DataFrame({
            'Tarih': [p.to_timestamp().strftime('%Y-%m-%d') for p in aylik_idx],
            'TMO_Giresun_TL_kg': tmo_giresun_series.values.round(2),
            'TMO_Levant_TL_kg':  tmo_levant_series.values.round(2),
            'Serbest_Piyasa_TL_kg': piyasa_series.values.round(2),
            'Sezon_Yili': [sezon_yili(p) for p in aylik_idx],
            'Hasat_Donemi': [1 if p.month in [8, 9, 10] else 0 for p in aylik_idx],
        })

        logger.info(f"Toplam {len(df)} aylık satır oluşturuldu.")
        return df

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek veri bulunamadı.")
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir  = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "turkiye_findik_fiyatlari.csv")

        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"Hedef değişken verisi kaydedildi: {file_name}")
        logger.info(f"\nSon 6 ay:\n{df.tail(6).to_string(index=False)}")


if __name__ == "__main__":
    logger.info("Fındık Fiyat Verisi Boru Hattı Başlatıldı...")
    builder = FindikFiyatBuilder()
    df = builder.build()
    builder.save_to_csv(df)
    logger.info("İşlem Tamamlandı.")

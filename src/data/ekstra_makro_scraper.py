import os
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EkstraMakroBuilder:
    """
    Fındık fiyatlarını dolaylı yoldan etkileyen 4 kritik "gizli" makro faktörü oluşturur:
    1. TCMB Politika Faizi (%) -> Fırsat maliyeti (parayı fındıkta mı faizde mi tutmalı?)
    2. Seçim Ekonomisi İndeksi -> Hükümetin TMO aracılığıyla taban fiyatı şişirme ihtimali
    3. Ramazan Bayramı Etkisi -> Çikolata/şekerleme talep patlaması ayı
    4. Baltic Dry Index (BDI) Proxy -> Küresel nakliye ve lojistik maliyeti
    
    Aylık formatta (2013-2026) çıktı üretir.
    """

    def build(self):
        logger.info("Ekstra makro ve sosyolojik göstergeler oluşturuluyor...")

        bugun = datetime.now()
        aylik_idx = pd.period_range(start='2013-01', end=f'{bugun.year}-12', freq='M')
        
        df = pd.DataFrame({'Yil_Ay': aylik_idx})
        df['Yil'] = df['Yil_Ay'].dt.year
        df['Ay'] = df['Yil_Ay'].dt.month
        df['Yil_Ay_Str'] = df['Yil_Ay'].astype(str)

        # ── 1. RAMAZAN BAYRAMI AYLARI (Hareketli Tatil Etkisi) ──
        # Ramazan bayramı her yıl ~11 gün geri gelir. Bayramın olduğu ve bir önceki ay (üretim hazırlığı) talebin zirvesidir.
        ramazan_aylari = {
            2013: 8, 2014: 7, 2015: 7, 2016: 7, 2017: 6, 2018: 6,
            2019: 6, 2020: 5, 2021: 5, 2022: 5, 2023: 4, 2024: 4,
            2025: 3, 2026: 3
        }
        df['Ramazan_Etkisi'] = df.apply(
            lambda r: 1 if r['Ay'] == ramazan_aylari.get(r['Yil']) or r['Ay'] == ramazan_aylari.get(r['Yil'])-1 else 0, 
            axis=1
        )

        # ── 2. SEÇİM EKONOMİSİ (1 = O yıl seçim var, fiyatlar desteklenir) ──
        # Yerel, Genel, Referandum yılları (Özellikle Ağustos TMO açıklamasını etkiler)
        secim_yillari = [2014, 2015, 2017, 2018, 2019, 2023, 2024]
        df['Secim_Gundemi'] = df['Yil'].apply(lambda y: 1 if y in secim_yillari else 0)

        # ── 3. BALTIC DRY INDEX (BDI) - Yıllık Ortalama Nakliye Maliyeti Proxy ──
        # Pandemi döneminde fırlayan nakliye maliyetleri ihracatçının alış fiyatını kırmasına neden oldu.
        bdi_yillik = {
            2013: 1205, 2014: 1105, 2015: 718, 2016: 673, 2017: 1145, 
            2018: 1352, 2019: 1352, 2020: 1066, 2021: 2943, 2022: 1934, 
            2023: 1378, 2024: 1800, 2025: 1450, 2026: 1400
        }
        df['BDI_Endeksi'] = df['Yil'].map(bdi_yillik)

        # ── 4. TCMB POLİTİKA FAİZİ (%) ──
        # Tüccarın fındığı stokta tutma fırsat maliyeti. Yüksek faiz = stok boşaltma baskısı = fiyat düşüşü.
        tcmb_faiz = {
            '2013-01': 5.5, '2014-01': 10.0, '2015-01': 8.25, '2016-01': 7.5, 
            '2017-01': 8.0, '2018-01': 8.0, '2018-09': 24.0, '2019-07': 19.75,
            '2019-12': 12.0, '2020-05': 8.25, '2020-12': 17.0, '2021-03': 19.0,
            '2021-12': 14.0, '2022-11': 9.0, '2023-06': 15.0, '2023-12': 42.5,
            '2024-03': 50.0, '2025-01': 50.0, '2026-01': 45.0
        }
        
        faiz_series = pd.Series(index=df['Yil_Ay_Str'], dtype=float)
        for tarih, faiz in tcmb_faiz.items():
            if tarih in faiz_series.index:
                faiz_series[tarih] = faiz
                
        # Boş ayları önceki aydan doldur (forward fill)
        faiz_series = faiz_series.ffill().bfill()
        df['TCMB_Faiz_Orani'] = faiz_series.values

        # Gereksiz geçici kolonları at
        df = df[['Yil_Ay_Str', 'TCMB_Faiz_Orani', 'BDI_Endeksi', 'Secim_Gundemi', 'Ramazan_Etkisi']]
        df.columns = ['Yil_Ay', 'TCMB_Faiz_Orani', 'BDI_Endeksi', 'Secim_Gundemi', 'Ramazan_Etkisi']
        
        return df

    def save_to_csv(self, df):
        if df is None or df.empty:
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(base_dir, "data", "raw")
        proc_dir = os.path.join(base_dir, "data", "processed")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(proc_dir, exist_ok=True)
        
        # Raw ve Processed klasörüne direk atalım (Temizleme gerektiren bir veri değil zaten)
        file_name_raw = os.path.join(raw_dir, "ekstra_makro_Sosyolojik.csv")
        file_name_proc = os.path.join(proc_dir, "ekstra_makro_Sosyolojik_temiz.csv")
        
        df.to_csv(file_name_raw, index=False, encoding='utf-8-sig')
        df.to_csv(file_name_proc, index=False, encoding='utf-8-sig')
        logger.info(f"Ekstra makro/sosyolojik özellikler kaydedildi: {file_name_proc}")


if __name__ == "__main__":
    builder = EkstraMakroBuilder()
    df = builder.build()
    builder.save_to_csv(df)
    logger.info("Ekstra özellikler başarıyla oluşturuldu!")

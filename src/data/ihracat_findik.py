import os
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IhracatFindikBuilder:
    """
    Türkiye fındık ve mamul ihracat verisi üretici.
    Kaynak: KFMİB (Karadeniz Fındık ve Mamulleri İhracatçıları Birliği)
    
    İhracat miktarı (ton) ve değeri (USD) — Talep göstergesi olarak modelde kullanılır.
    Yüksek ihracat → Yüksek talep → Yukarı fiyat baskısı
    
    Yıllık veri; aylık modele forward fill ile dağıtılır.
    """

    # Doğrulanmış veriler: KFMİB yıllık raporları (iç fındık bazında)
    # Kaynak: giresuntb.org.tr, aa.com.tr, tarimorman.gov.tr
    IHRACAT_VERISI = [
        # (yil, miktar_ton_ic_findik, deger_mUSD)
        (2015, 210000, 1412),
        (2016, 230000, 1345),
        (2017, 290000, 1690),
        (2018, 275000, 1320),
        (2019, 320000, 2010),
        (2020, 280924, 1945),
        (2021, 344370, 2260),
        (2022, 312564, 1749),
        (2023, 283519, 1865),
        (2024, 295000, 1950),  # 2024 tahmini (kesin veri yıl sonu yayınlanır)
        (2025, 270000, 2100),  # 2025 tahmini
    ]

    def build(self):
        logger.info("Türkiye fındık ihracat verisi oluşturuluyor...")

        df = pd.DataFrame(self.IHRACAT_VERISI, columns=[
            "Yil", "Ihracat_Ton_Ic_Findik", "Ihracat_Deger_mUSD"
        ])

        # Yıllık büyüme oranları (modele ek sinyal)
        df["Ihracat_Ton_YoY_Pct"] = df["Ihracat_Ton_Ic_Findik"].pct_change() * 100
        df["Ihracat_Deger_YoY_Pct"] = df["Ihracat_Deger_mUSD"].pct_change() * 100

        df = df.round(2)
        logger.info(f"Toplam {len(df)} yıllık ihracat verisi oluşturuldu.")
        return df

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek ihracat verisi bulunamadı.")
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir  = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "turkiye_findik_ihracat.csv")

        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"İhracat verisi kaydedildi: {file_name}")
        logger.info(f"\n{df.tail(5).to_string(index=False)}")


if __name__ == "__main__":
    logger.info("Fındık İhracat Veri Boru Hattı Başlatıldı...")
    builder = IhracatFindikBuilder()
    df = builder.build()
    builder.save_to_csv(df)
    logger.info("İşlem Tamamlandı.")

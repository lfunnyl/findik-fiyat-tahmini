import os
import requests
import pandas as pd
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TUFEScraper:
    """
    World Bank açık API'sinden Türkiye TÜFE (Enflasyon) verisini çeker.
    Endpoint: https://api.worldbank.org/v2/country/TR/indicator/FP.CPI.TOTL.ZG

    Yıllık enflasyon oranı (%) — aylık modelde forward fill ile yayılır.
    TÜİK'in kendi API'si güvenilmez ve değişken olduğundan World Bank tercih edildi.
    """

    URL = "https://api.worldbank.org/v2/country/TR/indicator/FP.CPI.TOTL.ZG"

    def fetch(self, start_yil=2015):
        logger.info(f"World Bank'tan Türkiye TÜFE verisi çekiliyor ({start_yil}–günümüz)...")

        bitis_yil = datetime.now().year
        params = {
            "format": "json",
            "date":   f"{start_yil}:{bitis_yil}",
            "per_page": 100,
        }

        try:
            response = requests.get(self.URL, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            kayitlar = data[1] if len(data) > 1 else []
            gozlemler = [
                {"Yil": int(r["date"]), "TUFE_Yillik_Pct": round(float(r["value"]), 2)}
                for r in kayitlar if r.get("value") is not None
            ]

            if not gozlemler:
                raise ValueError("Boş yanıt geldi.")

            df = pd.DataFrame(gozlemler).sort_values("Yil")
            logger.info(f"World Bank'tan {len(df)} yıllık TÜFE verisi alındı.")
            return df

        except Exception as e:
            logger.error(f"World Bank API hatası: {e} — Bilinen değerler kullanılıyor.")
            return self._fallback(start_yil)

    def _fallback(self, start_yil):
        """World Bank erişilemezse bilinen TÜİK bazlı yıllık TÜFE değerleri."""
        bilinen = {
            2015: 7.67, 2016: 7.78, 2017: 11.14, 2018: 20.30,
            2019: 15.18, 2020: 12.28, 2021: 19.60, 2022: 72.31,
            2023: 64.77, 2024: 52.00, 2025: 38.00,  # 2025 tahmini
        }
        donusum = [{"Yil": y, "TUFE_Yillik_Pct": v}
                   for y, v in bilinen.items() if y >= start_yil]
        return pd.DataFrame(donusum).sort_values("Yil")

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek TÜFE verisi bulunamadı.")
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir  = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "turkiye_tufe.csv")

        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"TÜFE verisi kaydedildi: {file_name}")
        logger.info(f"\n{df.tail(6).to_string(index=False)}")


if __name__ == "__main__":
    logger.info("TÜFE Veri Boru Hattı Başlatıldı...")
    scraper = TUFEScraper()
    df = scraper.fetch(start_yil=2015)
    scraper.save_to_csv(df)
    logger.info("İşlem Tamamlandı.")

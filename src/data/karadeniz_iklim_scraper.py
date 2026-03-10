import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KaradenizIklimScraper:
    """
    Türkiye'nin fındık kuşağındaki 3 ana bölgenin iklim verilerini çeker ve ortalar.

    Bölgeler:
      - Giresun  (40.91, 38.39) — Dünya kalite standartları merkezi
      - Ordu     (40.98, 37.88) — En büyük üretim hacmi
      - Trabzon  (41.00, 39.72) — Doğu Karadeniz üretim kuşağı

    API: Open-Meteo Historical Archive (ücretsiz, kayıt gerektirmez)
    Günlük veri:
      - Maks/Min sıcaklık → Don riski tespiti (Nisan-Mayıs çiçek dönemi kritik!)
      - Yağış miktarı → Hastalık (külleme) ve verim ilişkisi

    Hasat: Ağustos–Ekim arası (düzeltildi)
    """

    URL = "https://archive-api.open-meteo.com/v1/archive"

    BOLGELER = {
        "Giresun": (40.91, 38.39),
        "Ordu":    (40.98, 37.88),
        "Trabzon": (41.00, 39.72),
    }

    def __init__(self, yil_sayisi=5):
        self.end_date   = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=yil_sayisi * 365)).strftime('%Y-%m-%d')

    def _bolge_cek(self, bolge_adi, lat, lon):
        logger.info(f"{bolge_adi} iklim verisi çekiliyor ({self.start_date} → {self.end_date})...")
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": self.start_date,
            "end_date":   self.end_date,
            "daily":      ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone":   "auto",
        }
        try:
            response = requests.get(self.URL, params=params, timeout=20)
            response.raise_for_status()
            daily = response.json().get("daily", {})
            return pd.DataFrame({
                "Tarih":         daily.get("time", []),
                f"MaxC_{bolge_adi}": daily.get("temperature_2m_max", []),
                f"MinC_{bolge_adi}": daily.get("temperature_2m_min", []),
                f"Ygis_{bolge_adi}": daily.get("precipitation_sum", []),
            })
        except Exception as e:
            logger.error(f"{bolge_adi} verisi çekilemedi: {e}")
            return None

    def fetch_ortalama(self):
        """3 bölgenin verisini çekip günlük ortalama alır."""
        bolge_dfleri = []

        for adi, (lat, lon) in self.BOLGELER.items():
            df_b = self._bolge_cek(adi, lat, lon)
            if df_b is not None:
                bolge_dfleri.append(df_b)

        if not bolge_dfleri:
            logger.error("Hiçbir bölgeden veri çekilemedi!")
            return None

        # Tarihe göre birleştir
        df_merge = bolge_dfleri[0]
        for df_b in bolge_dfleri[1:]:
            df_merge = pd.merge(df_merge, df_b, on="Tarih", how="outer")

        bolgeler = list(self.BOLGELER.keys())

        # 3 ilçenin ortalaması
        df_merge["Max_Sicaklik_C"]   = df_merge[[f"MaxC_{b}" for b in bolgeler]].mean(axis=1).round(2)
        df_merge["Min_Sicaklik_C"]   = df_merge[[f"MinC_{b}" for b in bolgeler]].mean(axis=1).round(2)
        df_merge["Yagis_mm"]         = df_merge[[f"Ygis_{b}" for b in bolgeler]].mean(axis=1).round(2)

        # Tarımsal sinyal feature'ları
        df_merge["Don_Riski"]        = (df_merge["Min_Sicaklik_C"] < 0).astype(int)   # Genel don
        df_merge["Kritik_Don"]       = (                                                # Nisan-Mayıs = çiçek dönemi
            (df_merge["Min_Sicaklik_C"] < 0) &
            (pd.to_datetime(df_merge["Tarih"]).dt.month.isin([4, 5]))
        ).astype(int)
        df_merge["Hasat_Donemi"]     = (                                                # Ağustos-Ekim = hasat
            pd.to_datetime(df_merge["Tarih"]).dt.month.isin([8, 9, 10])
        ).astype(int)
        df_merge["Asgari_Yagis"]     = (df_merge["Yagis_mm"] > 50).astype(int)         # Aşırı yağış riski

        # Sadece özet + hesaplanan sütunları tut
        df_son = df_merge[["Tarih", "Max_Sicaklik_C", "Min_Sicaklik_C", "Yagis_mm",
                            "Don_Riski", "Kritik_Don", "Hasat_Donemi", "Asgari_Yagis"]].copy()

        logger.info(f"Toplam {len(df_son)} günlük iklim kaydı oluşturuldu (3 bölge ortalaması).")
        return df_son

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek iklim verisi bulunamadı.")
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir  = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "karadeniz_iklim_5_years.csv")

        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"İklim verisi kaydedildi: {file_name}")
        logger.info(f"Önizleme:\n{df.head(3).to_string(index=False)}")
        logger.info(f"Kritik don günleri: {df['Kritik_Don'].sum()} (Nisan-Mayıs)")


if __name__ == "__main__":
    logger.info("Karadeniz İklim Verisi Boru Hattı Başlatıldı...")
    scraper = KaradenizIklimScraper(yil_sayisi=5)
    df = scraper.fetch_ortalama()
    scraper.save_to_csv(df)
    logger.info("İşlem Tamamlandı.")

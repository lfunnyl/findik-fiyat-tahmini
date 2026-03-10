import os
import requests
import pandas as pd
import logging
from datetime import datetime, timedelta

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClimateDataScraper:
    def __init__(self):
        # Samsun / Çarşamba ovası yaklaşık koordinatları (Fındığın kalbi)
        self.lat = 41.20
        self.lon = 36.62
        
        # Son 5 yılın (5 * 365 gün) verisini alalım
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        # Open-Meteo Historical API Adresi
        self.url = "https://archive-api.open-meteo.com/v1/archive"
        
    def fetch_historical_weather(self):
        logger.info(f"Samsun/Çarşamba için {self.start_date} ile {self.end_date} arası iklim verisi çekiliyor...")
        
        # API'ye göndereceğimiz parametreler
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "daily": ["temperature_2m_max", "temperature_2m_min", "precipitation_sum"],
            "timezone": "auto"
        }
        
        try:
            response = requests.get(self.url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Gelen JSON verisinin içindeki 'daily' kısmını Pandas tablosuna çeviriyoruz
            daily_data = data.get('daily', {})
            df = pd.DataFrame({
                'Tarih': daily_data.get('time', []),
                'Max_Sicaklik_C': daily_data.get('temperature_2m_max', []),
                'Min_Sicaklik_C': daily_data.get('temperature_2m_min', []), # Don olayını buradan yakalayacağız
                'Yagis_Miktari_mm': daily_data.get('precipitation_sum', [])
            })
            
            return df
            
        except Exception as e:
            logger.error(f"İklim verisi çekilirken hata oluştu: {e}")
            return None

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek iklim verisi bulunamadı.")
            return

        # Sizin yazdığınız o mükemmel dinamik dosya yolu mimarisi!
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        file_name = os.path.join(raw_dir, "samsun_iklim_5_years.csv")
        
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"İklim verisi başarıyla kaydedildi: {file_name}")
        logger.info(f"Toplam {len(df)} günlük hava durumu verisi çekildi! 🌩️")
        
        # Verinin sadece ilk birkaç satırını ekrana basalım
        logger.info(f"Veri Önizleme:\n{df.head().to_string(index=False)}")

if __name__ == "__main__":
    logger.info("İklim Verisi Boru Hattı Başlatıldı...")
    scraper = ClimateDataScraper()
    df_weather = scraper.fetch_historical_weather()
    scraper.save_to_csv(df_weather)
    logger.info("İşlem Tamamlandı.")
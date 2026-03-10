import os
import yfinance as yf
import pandas as pd
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HistoricalUSDScraper:
    def __init__(self, years=13):
        # Yahoo Finance'de USD/TRY paritesinin kodu "TRY=X" tir.
        self.ticker = "TRY=X" 
        self.period = f"{years}y"
        
    def fetch_data(self):
        logger.info(f"Yahoo Finance üzerinden son {self.period} yıllık USD/TRY verisi çekiliyor...")
        try:
            # yfinance ile günlük (1d) veriyi indiriyoruz
            data = yf.download(self.ticker, period=self.period, interval="1d")
            
            if data.empty:
                logger.warning("Veri çekilemedi. İnternet bağlantınızı kontrol edin.")
                return None
            
            # İndeksi sıfırlayıp Tarih kolonunu normal bir kolona dönüştürüyoruz
            data = data.reset_index()
            
            # yfinance bazen çoklu indeks (MultiIndex) döner, bunu temizleyelim
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
                
            # Bize sadece Tarih ve Kapanış (Close) fiyatı lazım
            df = data[['Date', 'Close']].copy()
            df.columns = ['Tarih', 'USD_TRY_Kapanis']
            
            # Tarih formatını veritabanı standartlarına uygun hale getirelim (YIL-AY-GÜN)
            df['Tarih'] = df['Tarih'].dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            logger.error(f"Geçmiş veri çekilirken kritik hata: {e}")
            return None

    def save_to_csv(self, df):
        if df is None or df.empty:
            return

        # Proje ana dizinini bul ve raw klasörünün olduğundan emin olalım
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "historical_usd_try_5_years.csv")
        
        # Veriyi CSV olarak kaydedelim
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"Geçmiş veri başarıyla kaydedildi: {file_name}")
        logger.info(f"Toplam {len(df)} günlük döviz kuru verisi çekildi! 🚀")
        
        # Ekrana başından ve sonundan küçük bir önizleme basalım
        logger.info(f"İlk 3 Kayıt (En Eski):\n{df.head(3).to_string(index=False)}")
        logger.info(f"Son 3 Kayıt (En Yeni):\n{df.tail(3).to_string(index=False)}")

if __name__ == "__main__":
    logger.info("Geçmiş Döviz Verisi Boru Hattı Başlatıldı...")
    scraper = HistoricalUSDScraper(years=13)
    df_history = scraper.fetch_data()
    scraper.save_to_csv(df_history)
    logger.info("İşlem Tamamlandı.")

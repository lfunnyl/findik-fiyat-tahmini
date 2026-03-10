import os
import yfinance as yf
import pandas as pd
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MacroDataScraper:
    def __init__(self, years=13):
        # Tarımsal maliyet triumviratı:
        # 1. BZ=F  (Brent Petrol)   → Mazot, nakliye, gübre üretim maliyeti
        # 2. GC=F  (Altın Ons)      → Çiftçi güvenli limanı, enflasyon göstergesi
        # 3. UNG   (Natural Gas ETF) → Doğalgaz = Azotlu gübre üretiminin hammaddesi
        #    (Avrupa'da azotlu gübre, Rus doğalgazıyla yapılır — fındığa direk etkisi var)
        self.tickers = {
            "Brent_Petrol": "BZ=F",
            "Altin_Ons":    "GC=F",
            "Dogalgaz_ETF": "UNG",   # United States Natural Gas Fund (gübre maliyeti proxy)
        }
        self.period = f"{years}y"
        
    def fetch_data(self):
        logger.info(f"Yfinance üzerinden son {self.period} yıllık makroekonomik veriler çekiliyor...")
        
        all_data = pd.DataFrame()
        
        try:
            for name, ticker in self.tickers.items():
                logger.info(f"{name} ({ticker}) verisi indiriliyor...")
                data = yf.download(ticker, period=self.period, interval="1d")
                
                if data.empty:
                    logger.warning(f"{name} verisi çekilemedi.")
                    continue
                
                # İndeksi sıfırla
                data = data.reset_index()
                
                # MultiIndex dönerse formatı düzelt
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)
                    
                # Tarih ve Kapanışı al
                df_temp = data[['Date', 'Close']].copy()
                df_temp.columns = ['Tarih', f'{name}_Kapanis']
                df_temp['Tarih'] = df_temp['Tarih'].dt.strftime('%Y-%m-%d')
                
                # Verileri Tarih bazında birleştir (Merge)
                if all_data.empty:
                    all_data = df_temp
                else:
                    all_data = pd.merge(all_data, df_temp, on='Tarih', how='outer')
            
            # Tarihe göre sırala ve tatillerden/haftasonlarından kaynaklı boşlukları önceki günkü veriyle doldur (Forward Fill)
            all_data = all_data.sort_values('Tarih').ffill()
            
            return all_data
            
        except Exception as e:
            logger.error(f"Makro veri çekilirken hata: {e}")
            return None

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek makro veri bulunamadı.")
            return

        # Sağlam yapı: Dinamik Dizin
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "makro_veriler_5_years.csv")
        
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"Makro veriler başarıyla kaydedildi: {file_name}")
        logger.info(f"Toplam {len(df)} günlük satır oluşturuldu. 🌍")
        logger.info(f"Son Günlerin Önizlemesi:\n{df.tail(3).to_string(index=False)}")

if __name__ == "__main__":
    logger.info("Makro Veri Boru Hattı Başlatıldı...")
    scraper = MacroDataScraper(years=13)
    df_macro = scraper.fetch_data()
    scraper.save_to_csv(df_macro)
    logger.info("İşlem Tamamlandı.")

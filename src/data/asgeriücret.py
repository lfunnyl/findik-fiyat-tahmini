import os
import pandas as pd
import logging
from datetime import datetime

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MacroDataBuilder:
    def __init__(self):
        # 2013'ten 2026'ya kadar net asgari ücret (TL/ay) değişim tarihleri
        # Kaynak: Aile ve Sosyal Hizmetler Bakanlığı, Çalışma Bakanlığı kararnameleri
        self.wage_data = {
            # ── 2013-2020: Tarihi veriler ──
            '2013-01-01':  846.90,
            '2014-01-01': 1071.00,
            '2015-01-01': 1000.74,   # Net asgari ücret (büyük zam yılı)
            '2016-01-01': 1300.99,
            '2017-01-01': 1404.06,
            '2018-01-01': 1603.12,
            '2018-07-01': 2020.90,   # Yıl içi ara zam
            '2019-01-01': 2558.40,
            '2020-01-01': 2942.55,
            # ── 2021-2026: Yüksek enflasyon dönemi ──
            '2021-01-01': 2825.90,
            '2022-01-01': 4253.40,
            '2022-07-01': 5500.35,
            '2023-01-01': 8506.80,
            '2023-07-01': 11402.32,
            '2024-01-01': 17002.12,
            '2025-01-01': 22002.00,
            '2026-01-01': 28000.00,
        }
        
    def build_daily_macro_data(self):
        logger.info("Makroekonomik referans verisi oluşturuluyor...")
        
        try:
            # Önce sözlüğü bir Pandas tablosuna çevirelim
            df = pd.DataFrame(list(self.wage_data.items()), columns=['Tarih', 'Asgari_Ucret_TL'])
            df['Tarih'] = pd.to_datetime(df['Tarih'])
            df.set_index('Tarih', inplace=True)
            
            # Asıl Sihir Burada: Veriyi günlüğe çevirip boşlukları "İleri Doğru" dolduruyoruz (Forward Fill)
            # Böylece örneğin 15 Mart 2023'teki asgari ücreti sorduğumuzda 8506 TL olarak verecek.
            bugun = datetime.now().strftime('%Y-%m-%d')
            # 2013'ten bugüne kadar her gün için bir satır oluştur
            daily_index = pd.date_range(start='2013-01-01', end=bugun, freq='D')
            
            # Tabloyu bu yeni günlük indekse oturt ve boşlukları bir önceki geçerli değerle doldur
            df_daily = df.reindex(daily_index).ffill()
            
            # İndeksi tekrar normal kolona alalım
            df_daily = df_daily.reset_index()
            df_daily.columns = ['Tarih', 'Asgari_Ucret_TL']
            df_daily['Tarih'] = df_daily['Tarih'].dt.strftime('%Y-%m-%d')
            
            return df_daily
            
        except Exception as e:
            logger.error(f"Makro veri oluşturulurken hata: {e}")
            return None

    def save_to_csv(self, df):
        if df is None or df.empty:
            return

        # Sizin yazdığınız o harika dinamik dosya yolu!
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        file_name = os.path.join(raw_dir, "turkiye_asgeri_ücret_veri.csv")
        
        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"Makro veri başarıyla kaydedildi: {file_name}")
        logger.info(f"Toplam {len(df)} günlük asgari ücret haritası çıkarıldı! 💼")
        logger.info(f"Veri Önizleme:\n{df.tail().to_string(index=False)}")

if __name__ == "__main__":
    logger.info("Makro Veri Boru Hattı Başlatıldı...")
    builder = MacroDataBuilder()
    df_macro = builder.build_daily_macro_data()
    builder.save_to_csv(df_macro)
    logger.info("İşlem Tamamlandı.")
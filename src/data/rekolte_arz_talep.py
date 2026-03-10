import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RekolteArzTalepBuilder:
    """
    Türkiye ve dünya fındık piyasasının arz-talep tablosunu oluşturur.

    İçerik:
    1. Rekolte Tahmini vs Gerçekleşen
       - Ön-sezon tahmini (Mayıs-Haziran): Tarım Bakanlığı / Fiskobirlik açıklamaları
       - Gerçekleşen üretim: FAO/TMO nihai rakamları
       - Tahmini sapma (surprize etkisi) → fiyat volatilitesinin ana kaynağı

    2. Randıman Kalitesi
       - TMO '% sağlam iç' oranı → yüksek randıman = fazla arz = düşük fiyat baskısı
       - Bilinen sezon randımanları (TMO açıklamalarından)

    3. Küresel Arz-Talep Dengesi
       - Türkiye + İtalya + Azerbaycan = Dünya arzının %85'i
       - Küresel değer-fındık tüketimi (FAOSTAT Trade data)
       - Stok-kullanım oranı (proxy)

    Hepsi yıllık. Aylık tabloya forward fill ile dağıtılır.
    """

    # ── 1. Rekolte: Ön-Sezon Tahmini vs Gerçekleşen (ton, Türkiye) ──────────
    # Kaynak: Fiskobirlik Mayıs-Haziran bültenleri, Tarım Bakanlığı, Reuters
    # Tahmini sapma (surprise) = Gerçekleşen / Tahmin - 1 → pozitif = beklenenden iyi
    REKOLTE = {
        # sezon_yili: (on_sezon_tahmin_ton, gerceklesen_ton)
        2015: (700_000, 646_000),
        2016: (500_000, 420_000),   # Kış kondisyonu kötü, büyük negatif sürpriz → fiyat patladı
        2017: (600_000, 675_000),   # Pozitif sürpriz → fiyat baskısı
        2018: (550_000, 537_000),
        2019: (700_000, 773_000),   # Rekor yıl, pozitif sürpriz
        2020: (700_000, 710_000),
        2021: (700_000, 750_000),   # Hafif iyi sezon
        2022: (750_000, 710_000),   # Negatif sürpriz (beklentiden düşük) → fiyat baskısı
        2023: (680_000, 647_000),   # Negatif sürpriz
        2024: (700_000, 700_000),   # Tahmini
        2025: (680_000, 620_000),   # Don hasarı sonrası revize (tahmini)
    }

    # ── 2. Randıman Kalitesi (% 50 sağlam iç esasına göre gerçek oran) ───────
    # Kaynak: TMO alım raporları, Fiskobirlik teknik bültenleri
    # %50 = standart; > 50 yüksek kalite (fazla arz), < 50 don/kuraklık hasarı
    RANDIMAN = {
        # sezon_yili: ortalama_randiman_pct
        2015: 51.5,
        2016: 48.0,   # Kötü sezon, düşük randıman
        2017: 52.5,
        2018: 50.5,
        2019: 53.0,   # Rekor kalite
        2020: 51.0,
        2021: 51.5,
        2022: 50.0,
        2023: 49.5,   # Hafif don hasarı etkisi
        2024: 50.5,
        2025: 47.0,   # Mart 2025 don hasarı — ciddi randıman düşüşü
    }

    # ── 3. Küresel Arz-Talep Dengesi ─────────────────────────────────────────
    # Kaynak: INC (International Nut & Dried Fruit Council), FAOSTAT Trade
    # Dünya tüketimi ≈ dünya üretimi + geçen yıl stoku - bu yıl stoku
    GLOBAL = {
        # sezon_yili: (dunya_uretim_ton, tahmini_tuketim_ton, italya_uretim, azerbaycan_uretim)
        2015: (920_000,  880_000, 87_000, 30_000),
        2016: (700_000,  870_000, 75_000, 32_000),  # Büyük açık → fiyat artışı
        2017: (970_000,  890_000, 79_000, 35_000),
        2018: (820_000,  880_000, 89_000, 37_000),
        2019: (1_080_000, 920_000, 65_000, 39_000),
        2020: (1_020_000, 950_000, 72_000, 43_000),
        2021: (1_060_000, 980_000, 73_000, 46_000),
        2022: (1_000_000, 990_000, 72_000, 51_000),
        2023: (940_000, 1_000_000, 70_000, 48_000),  # İlk kez tüketim > üretim!
        2024: (1_000_000, 1_010_000, 68_000, 45_000),
        2025: (960_000, 1_020_000, 65_000, 50_000),   # Tahmini
    }

    def build(self):
        logger.info("Rekolte / Arz-Talep / Randıman verisi oluşturuluyor...")
        satirlar = []

        for sezon, (tahmin, gercek) in self.REKOLTE.items():
            randiman = self.RANDIMAN.get(sezon, 50.0)
            global_v  = self.GLOBAL.get(sezon, (None, None, None, None))
            dunya_uretim, dunya_tuketim, italya, azerbaycan = global_v

            # ── Türkiye arz dengesi ──
            tr_ihracat_tahmini = gercek * 0.70  # Türkiye geleneksel olarak ~%70 ihraç eder

            # ── Küresel göstergeler ──
            global_arz_acigi  = None
            stok_kullanim     = None
            tr_pazar_payi     = None

            if dunya_uretim and dunya_tuketim:
                global_arz_acigi = dunya_tuketim - dunya_uretim   # pozitif = arz yetersiz → fiyat ↑
                stok_kullanim    = round(dunya_uretim / dunya_tuketim * 100, 2)
                tr_pazar_payi    = round(gercek / dunya_uretim * 100, 2) if dunya_uretim else None

            satirlar.append({
                "Sezon_Yili":            sezon,
                # Rekolte
                "TR_Rekolte_Tahmin_Ton": tahmin,
                "TR_Rekolte_Gercek_Ton": gercek,
                "Rekolte_Surprise_Pct":  round((gercek / tahmin - 1) * 100, 2),   # + = iyi, - = kötü
                "Rekolte_Sapma_Ton":     gercek - tahmin,
                # Randıman
                "Randiman_Pct":          randiman,
                "Randiman_Kalite":       "Yuksek" if randiman >= 51 else ("Normal" if randiman >= 49 else "Dusuk"),
                # Global arz-talep
                "Dunya_Uretim_Ton":      dunya_uretim,
                "Dunya_Tuketim_Ton":     dunya_tuketim,
                "Global_Arz_Acigi_Ton":  global_arz_acigi,   # pozitif → kıtlık sinyali
                "Stok_Kullanim_Oran_Pct": stok_kullanim,     # < 100 → fiyat baskısı yukarı
                "TR_Pazar_Payi_Pct":     tr_pazar_payi,
                "Italya_Uretim_Ton":     italya,
                "Azerbaycan_Uretim_Ton": azerbaycan,
            })

        df = pd.DataFrame(satirlar).sort_values("Sezon_Yili").reset_index(drop=True)

        # ── Lag özelliği: geçen yılın surprise'ı bu yılın fiyatını etkiler ──
        df["Rekolte_Surprise_Lag1"] = df["Rekolte_Surprise_Pct"].shift(1)

        # ── 3 yıllık hareketli ortalama üretim (trend göstergesi) ──
        df["TR_Uretim_MA3"] = df["TR_Rekolte_Gercek_Ton"].rolling(3, min_periods=1).mean().round(0)

        # ── Küresel kıtlık skalası: 0-100 normalize ──
        if df["Global_Arz_Acigi_Ton"].notna().any():
            min_v = df["Global_Arz_Acigi_Ton"].min()
            max_v = df["Global_Arz_Acigi_Ton"].max()
            rng   = max_v - min_v if max_v != min_v else 1
            df["Global_Kitlik_Skoru"] = ((df["Global_Arz_Acigi_Ton"] - min_v) / rng * 100).round(2)

        logger.info(f"Toplam {len(df)} sezonluk arz-talep verisi oluşturuldu.")
        return df

    def save_to_csv(self, df):
        if df is None or df.empty:
            logger.warning("Kaydedilecek veri bulunamadı.")
            return

        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_dir  = os.path.join(base_dir, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)
        file_name = os.path.join(raw_dir, "rekolte_arz_talep.csv")

        df.to_csv(file_name, index=False, encoding='utf-8-sig')
        logger.info(f"Rekolte & arz-talep verisi kaydedildi: {file_name}")
        logger.info(f"\n{df[['Sezon_Yili','TR_Rekolte_Tahmin_Ton','TR_Rekolte_Gercek_Ton','Rekolte_Surprise_Pct','Randiman_Pct','Global_Arz_Acigi_Ton','Stok_Kullanim_Oran_Pct']].to_string(index=False)}")


if __name__ == "__main__":
    logger.info("Rekolte & Arz-Talep Veri Boru Hattı Başlatıldı...")
    builder = RekolteArzTalepBuilder()
    df = builder.build()
    builder.save_to_csv(df)
    logger.info("İşlem Tamamlandı.")

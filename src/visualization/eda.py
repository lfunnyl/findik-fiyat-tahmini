import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Görselleştirme ayarları
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_data():
    filepath = os.path.join(PROC_DIR, "master_features.csv")
    if not os.path.exists(filepath):
        logger.error(f"Dosya bulunamadı: {filepath}")
        return None
    df = pd.read_csv(filepath)
    df['Tarih'] = pd.to_datetime(df['Tarih'])
    return df

def plot_time_series(df):
    """Fiyatların zaman içindeki değişimi."""
    logger.info("Zaman serisi grafikleri çiziliyor...")
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    color = 'tab:blue'
    ax1.set_xlabel('Tarih')
    ax1.set_ylabel('Fiyat (TL/kg)', color=color)
    ax1.plot(df['Tarih'], df['Serbest_Piyasa_TL_kg'], label='Serbest Piyasa Fiyatı', color=color, linewidth=2.5)
    ax1.plot(df['Tarih'], df['TMO_Giresun_TL_kg'], label='TMO Giresun Taban', color='tab:cyan', linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('USD/TRY Kuru', color=color)  
    ax2.plot(df['Tarih'], df['USD_TRY_Kapanis'], label='USD/TRY Kuru', color=color, alpha=0.6)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='lower right')
    
    plt.title('Fındık Serbest Piyasa Fiyatı ve USD/TRY Kurunun Gelişimi (2013-2026)')
    fig.tight_layout()  
    plt.savefig(os.path.join(REPORTS_DIR, '01_fiyat_zaman_serisi.png'), dpi=300)
    plt.close()

def plot_correlation_matrix(df):
    """Hedef değişken ile en yüksek korelasyona sahip öznitelikleri bulur."""
    logger.info("Korelasyon analizi yapılıyor...")
    
    # Sadece sayısal kolonları al
    numeric_df = df.select_dtypes(include=[np.number])
    
    # NaN değerleri korelasyon hesabı için geçici olarak düş veya doldur
    corr_matrix = numeric_df.corr()
    
    # Fiyat ile en yüksek korelasyona sahip 15 özellik
    target = 'Serbest_Piyasa_TL_kg'
    if target not in corr_matrix.columns:
        return
        
    top_correlations = corr_matrix[target].abs().sort_values(ascending=False).head(16).index
    top_corr_matrix = numeric_df[top_correlations].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(top_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(f'{target} ile En Yüksek Korelasyona Sahip 15 Özellik')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '02_korelasyon_matrisi.png'), dpi=300)
    plt.close()
    
    # Raporlama için en güçlü korelasyonları yazdır
    strong_corrs = corr_matrix[target].sort_values(ascending=False).drop(target)
    logger.info(f"En güçlü pozitif korelasyonlar:\n{strong_corrs.head(5)}")
    logger.info(f"En güçlü negatif korelasyonlar:\n{strong_corrs.tail(5)}")

def plot_scatter_relationships(df):
    """Fiyatı etkilediği varsayılan spesifik özelliklerin scatter plotları."""
    logger.info("Scatter plot analizleri yapılıyor...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. USD/TRY Etkisi
    sns.scatterplot(ax=axes[0, 0], x='USD_TRY_Kapanis', y='Serbest_Piyasa_TL_kg', data=df, hue='Yil', palette='viridis', alpha=0.8)
    axes[0, 0].set_title('Dolar Kuru vs Fındık Fiyatı')
    
    # 2. Asgari Ücret Etkisi (Maliyet)
    sns.scatterplot(ax=axes[0, 1], x='Asgari_Ucret_TL', y='Serbest_Piyasa_TL_kg', data=df, hue='Yil', palette='magma', alpha=0.8)
    axes[0, 1].set_title('Asgari Ücret (Maliyet) vs Fındık Fiyatı')
    
    # 3. İhracat Miktarı Etkisi (Talep)
    sns.scatterplot(ax=axes[1, 0], x='Ihracat_Ton_Ic_Findik', y='Serbest_Piyasa_TL_kg', data=df, hue='Yil', palette='cool', alpha=0.8)
    axes[1, 0].set_title('İhracat (Ton) vs Fındık Fiyatı')
    
    # 4. Fiyat_Lag1 (Otokorelasyon)
    sns.scatterplot(ax=axes[1, 1], x='Fiyat_Lag1', y='Serbest_Piyasa_TL_kg', data=df, alpha=0.6, color='darkgreen')
    axes[1, 1].set_title('Geçen Ayın Fiyatı vs Bu Ayın Fiyatı (Otokorelasyon)')
    axes[1, 1].plot([0, 300], [0, 300], 'r--', alpha=0.5) # y=x doğrusu
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '03_ikili_iliskiler.png'), dpi=300)
    plt.close()

def plot_harvest_quality_impact(df):
    """Rekolte sürprizi ve randımanın fiyat değişimine etkisi."""
    logger.info("Rekolte ve kalite etkisi analizi yapılıyor...")
    
    # Sadece hasat aylarında (Ağustos-Ekim) veya yılın genelinde ortalama fiyatı gösterilebilir
    # Yıllık bazda analiz yapalım
    df_yil = df.groupby('Sezon_Yili').agg({
        'Serbest_Piyasa_TL_kg': 'mean',
        'Fiyat_USD_kg': 'mean',
        'Rekolte_Surprise_Pct': 'first',
        'Randiman_Pct': 'first',
        'Global_Arz_Acigi_Ton': 'first'
    }).dropna()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Rekolte Sürprizi vs Dolar Bazlı Fiyat
    # Dolar bazlı fiyat, enflasyondan arındırılmış gerçek değerlemeyi gösterir
    sns.regplot(ax=axes[0], x='Rekolte_Surprise_Pct', y='Fiyat_USD_kg', data=df_yil, 
                scatter_kws={'s': 100, 'alpha': 0.7}, line_kws={'color': 'red', 'linestyle': '--'})
    axes[0].set_title('Rekolte Sürprizi (%) vs Gerçek Fiyat (USD/kg)')
    axes[0].set_xlabel('Rekolte Sürprizi (Negatif = Beklentiden Kötü Üretim)')
    axes[0].set_ylabel('Sezonluk Ortalama Fiyat (USD/kg)')
    
    # Yılları etiketle
    for idx, row in df_yil.iterrows():
        axes[0].annotate(str(idx), (row['Rekolte_Surprise_Pct'], row['Fiyat_USD_kg']), 
                         xytext=(5, 5), textcoords='offset points', fontsize=9)
                         
    # 2. Global Arz Açığı vs Randıman
    sns.scatterplot(ax=axes[1], x='Randiman_Pct', y='Fiyat_USD_kg', data=df_yil, 
                    size='Global_Arz_Acigi_Ton', sizes=(50, 400), hue='Global_Arz_Acigi_Ton', palette='viridis')
    axes[1].set_title('Randıman (%) & Global Arz Açığı Etkisi')
    axes[1].set_xlabel('Randıman Kalesi (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '04_rekolte_randiman_etkisi.png'), dpi=300)
    plt.close()


def plot_advanced_features(df):
    """Kullanıcının önerdiği gelişmiş 6 özelliğin fiyatla ilişkisini gösterir."""
    logger.info("Gelişmiş özellik analizleri yapılıyor...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Kakao
    sns.scatterplot(ax=axes[0, 0], x='Kakao_Kapanis', y='Serbest_Piyasa_TL_kg', data=df, hue='Yil', palette='copper', alpha=0.8)
    axes[0, 0].set_title('Kakao Fiyatı vs Fındık Fiyatı')
    
    # 2. Kur Volatilitesi
    sns.scatterplot(ax=axes[0, 1], x='Kur_Volatilite_30G', y='Serbest_Piyasa_TL_kg', data=df, color='crimson', alpha=0.7)
    axes[0, 1].set_title('Kur Volatilitesi (Belirsizlik) vs Fiyat')
    
    # 3. Seçim Gündemi
    sns.boxplot(ax=axes[0, 2], x='Secim_Gundemi', y='Serbest_Piyasa_TL_kg', data=df, palette='Set2')
    axes[0, 2].set_title('Seçim Gündemi Etkisi (0=Yok, 1=Seçim Yılı)')
    
    # 4. TCMB Faizi
    sns.scatterplot(ax=axes[1, 0], x='TCMB_Faiz_Orani', y='Serbest_Piyasa_TL_kg', data=df, hue='Yil', palette='plasma', alpha=0.8)
    axes[1, 0].set_title('TCMB Politika Faizi (%) vs Fiyat')
    
    # 5. Ramazan Etkisi
    sns.boxplot(ax=axes[1, 1], x='Ramazan_Etkisi', y='Serbest_Piyasa_TL_kg', data=df, palette='Set3')
    axes[1, 1].set_title('Ramazan/Bayram Etkisi (0=Normal, 1=Yüksek Talep)')
    
    # 6. BDI Endeksi (Lojistik)
    sns.scatterplot(ax=axes[1, 2], x='BDI_Endeksi', y='Serbest_Piyasa_TL_kg', data=df, hue='Yil', palette='Blues', alpha=0.8)
    axes[1, 2].set_title('Baltic Dry Index (Nakliye Maliyeti) vs Fiyat')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, '05_gelismis_ozellikler.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    logger.info("--- EDA BAŞLADI ---")
    df_master = load_data()
    if df_master is not None:
        plot_time_series(df_master)
        plot_correlation_matrix(df_master)
        plot_scatter_relationships(df_master)
        plot_harvest_quality_impact(df_master)
        plot_advanced_features(df_master)
        logger.info(f"--- EDA TAMAMLANDI. Grafikler '{REPORTS_DIR}' klasörüne kaydedildi. ---")
    else:
        logger.error("Veri yüklenemedi!")

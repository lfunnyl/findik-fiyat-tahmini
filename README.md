# 🌰 Türkiye Fındık Fiyat Tahmin Sistemi (End-to-End AI/ML)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-000000?logo=next.js&logoColor=white)
![Machine Learning](https://img.shields.io/badge/ML-Optuna%20%7C%20CatBoost%20%7C%20SHAP-FF6F00)

**Kurumsal Seviye Emtia Fiyat Tahmin ve Karar Destek Sistemi**  
*XGBoost · LightGBM · CatBoost · Delta Modeling · Causal Inference*

</div>

---

## 🌟 Proje Nedir?

Türkiye, dünya fındık üretiminin **~%70'ini** tek başına karşılayarak küresel piyasada tekel konumundadır. Ancak döviz kurlarındaki dalgalanmalar, iklim şokları (don olayları) ve devletin (TMO) müdahaleleri nedeniyle serbest piyasadaki fındık fiyatlarını öngörmek son derece zordur. Üreticiler ve ihracatçı şirketler için doğru fiyat tahmini, milyonlarca dolarlık kar/zarar anlamına gelir.

Bu proje, fındık fiyatlarını (Reel USD/kg bazında) **%5.06 MAPE** hata payıyla tahmin edebilen, yatırımcılara ve üreticilere veri odaklı (Data-Driven) bir karar destek mekanizması sunan **uçtan uca (End-to-End)** bir Yapay Zeka platformudur.

---

## 🏗️ Neler Yaptım? (Proje Kapsamı)

Bu sistemi sıfırdan canlıya (production) alırken şu adımları bizzat tasarladım ve kodladım:

1. **Veri Toplama (Data Engineering):** TCMB (Kur/Enflasyon), TMO (Taban fiyatlar), FAO (Küresel Rekolte) ve Open-Meteo (İklim/Don riski) gibi 14 farklı veri kaynağından 150+ aylık (2013-2026) veriyi otomatik çeken scraper'lar yazdım.
2. **Özellik Mühendisliği (Feature Engineering):** Sadece fiyatı değil; kur volatilitesini, hareketli ortalamaları (MA), TMO'nun piyasa ile olan fiyat makasını ve iklim şoklarını hesaplayan modüler bir veri boru hattı (Pipeline) kurdum.
3. **Makine Öğrenmesi (ML Modeling):** XGBoost, LightGBM ve CatBoost algoritmalarını Optuna ile eğittim. Modellerin "kara kutu" olmasını engellemek için **SHAP** entegrasyonu ile kararların şeffafça açıklanmasını sağladım.
4. **Full-Stack Mimari:** Geliştirdiğim modelleri bir Python script'i olmaktan çıkarıp, **FastAPI** ile bir Backend servisine bağladım. Kullanıcıların tahminleri görebilmesi için **Next.js** ve **TailwindCSS** kullanarak modern bir Dashboard (Frontend) tasarladım.

---

## 🧠 Karşılaştığım Krizler ve Çözümlerim (Lessons Learned)

Her şey kağıt üzerinde mükemmel başlasa da, gerçek dünya verisiyle uğraşırken ve sistemi sunucuya yüklerken ciddi problemlerle yüzleştim. İşte o problemler ve bulduğum mühendislik çözümleri:

### Kriz 1: Momentum (Random Walk) Tuzağı
* **Problem:** Makine öğrenmesi modellerimi ilk eğittiğimde R² skorum 0.95 çıkmıştı. Ancak detaylı analiz ettiğimde, modelin sadece $Y_t \approx Y_{t-1}$ yaptığını, yani "Bugünkü fiyat neyse yarınki de o olur" diyerek dünkü fiyatı kopyaladığını (tembellik yaptığını) fark ettim. Gerçek krizleri asla öngöremiyordu.
* **Çözüm (Delta Target Transformation):** Modelin hedefini (Target) mutlak fiyat olarak vermeyi bıraktım. Bunun yerine **Fiyatın Logaritmik Değişimini (Delta)** tahmin etmesini istedim. Model artık dünkü fiyatı kopyalayamıyor; rekolteye, kura ve enflasyona bakarak piyasanın *ne yöne hareket edeceğini* öğrenmek zorunda kalıyordu. 

### Kriz 2: Kayıp Nedensellik ve TMO Müdahalesi
* **Problem:** Ağustos 2025 gibi aylarda fiyatın bir anda %50 fırladığı dönemler vardı. Ne kurda ne de iklimde bir şok yokken yaşanan bu sıçramaları algoritmalar "anomali" diyerek eliyordu.
* **Çözüm (Causal Forcing):** Fındık piyasasında fiyatı asıl belirleyen şeyin devletin (TMO) yaptığı taban fiyat müdahaleleri olduğunu (Domain Knowledge) analize dahil ettim. TMO'nun güncel fiyatı ile serbest piyasa arasındaki makası (`TMO_Mevcut_Makas`) hesaplayarak modele **zorla (VIP Feature)** ekledim. Model artık devletin piyasayı yukarı çekeceğini önceden anlayabiliyordu.

### Kriz 3: Monolith'ten Mikroservise Geçiş ve Sunucu Çökmesi
* **Problem:** Projenin başında veri çekimi, tahmin ve ön yüz aynı dosya (`app.py`) içindeydi. Bu devasa yığını sunucuya (Railway) yüklemeye kalktığımda, ağır ML kütüphaneleri (PyTorch, Prophet vb.) yüzünden RAM sınırları aşılıyor ve sistem kilitleniyordu (Build Timeout).
* **Çözüm:** Sisteme "Refactoring" uyguladım. Feature Engineering mantığını `src/features/` altına izole ettim. Sadece canlı ortama (Production) özel, ağır kütüphaneleri dışlayan hafifletilmiş bir `requirements-api.txt` yazdım. FastAPI ve Next.js'i birbirinden tamamen ayırarak (Decoupling) derleme süresini dakikalardan 40 saniyeye düşürdüm.

---

## 📊 Model Sonuçları (Test Seti — Reel USD/kg)

Tüm bu optimizasyonlar sonucunda elde edilen güncel test metrikleri:

| Model | R² | MAE (USD/kg) | RMSE | MAPE |
|---|---|---|---|---|
| Ridge Baseline | 0.6437 | 0.452 | 0.647 | 8.82% |
| XGBoost (Optuna)| 0.7904 | 0.284 | 0.496 | 5.48% |
| LightGBM (Optuna)| **0.7991** | **0.273** | **0.486** | 5.30% |
| **CatBoost** | 0.7853 | 0.276 | 0.502 | **5.06%** |

---

## 🚀 Kurulum ve Çalıştırma

Proje mikroservis mimarisiyle tasarlanmıştır.

### Gereksinimler
- Python 3.11+
- Node.js 18+

### 1. ML Modellerini Eğitmek (Opsiyonel)
Modelleri baştan eğitmek ve SHAP metriklerini üretmek için:
```bash
python src/models/train_model.py
python scripts/generate_shap.py
```

### 2. FastAPI Backend'i Başlatmak
```bash
pip install -r requirements-api.txt
python api/main.py
```
*(Backend http://localhost:8000 adresinde çalışır)*

### 3. Next.js Dashboard'u Başlatmak
```bash
cd dashboard
npm install
npm run dev
```
*(Dashboard http://localhost:3000 adresinde çalışır)*

---

<div align="center">
  <sub>Developed with Advanced AI & Domain Knowledge · 2026</sub>
</div>

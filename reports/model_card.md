# 📋 Model Card — Fındık Fiyat Tahmin Sistemi

**Model Adı:** Weighted Ensemble (XGBoost + Ridge)  
**Versiyon:** 3.1.0  
**Tarih:** Nisan 2026  
**Yazar:** lfunnyl  

---

## 1. Model Amacı

Bu sistem, Türkiye serbest piyasa fındık fiyatlarını **1 ila 9 ay** önceden tahmin etmek için geliştirilmiştir. Temel kullanım senaryoları:

| Kullanıcı | Amaç |
|---|---|
| **Üretici** | Hasat öncesi fiyat beklentisi oluşturma |
| **İhracatçı** | Kontrat müzakeresi için fiyat bandı belirleme |
| **Politika yapıcı** | TMO taban fiyat önerisinin değerlendirilmesi |
| **Araştırmacı** | Kur-fiyat nedensel ilişkisinin analizi |

---

## 2. Veri Kaynakları

| Kaynak | Veri | Dönem | Format |
|---|---|---|---|
| Toprak Mahsulleri Ofisi (TMO) | Serbest piyasa + taban fiyatları | 2013–2026 | Aylık |
| TCMB | USD/TRY kur, TÜFE | 2013–2026 | Günlük/Aylık |
| FAO | Dünya fındık üretimi (Türkiye + diğer) | 2013–2024 | Yıllık |
| Open-Meteo API | Karadeniz iklim (yağış, sıcaklık, don) | 2018–2026 | Günlük → Aylık |
| yFinance | Brent petrol, altın, doğalgaz, kakao | 2013–2026 | Günlük → Aylık |
| BLS (ABD) | ABD CPI (reel USD dönüşümü) | 2013–2024 | Aylık |
| TÜİK | Asgari ücret, maliyet endeksi | 2013–2026 | Yıllık |
| Özel scraperlar | İhracat miktar/değer, rekolte tahminleri | 2013–2026 | Aylık |

**Toplam:** 152 satır × 60+ özellik (Nisan 2013 – Mart 2026)

---

## 3. Metodoloji

### 3.1 Hedef Değişken

Türk Lirası baz fiyat yerine **Reel USD/kg** kullanılmaktadır:

```
Fiyat_RealUSD_kg = (Serbest_Piyasa_TL_kg / USD_TRY_Kapanis) × (US_CPI_2024 / US_CPI_yil)
```

**Gerekçe:** TL bazlı fiyatlar Türkiye enflasyonuyla (2021–2024 döneminde %80+ TÜFE) kirlenmiştir. Reel USD hedefi, temporal covariate shift'i önemli ölçüde azaltır.

### 3.2 Walk-Forward Expanding Window CV

```
Fold 1: ████░░░░░░░  (train | test)
Fold 2: ██████░░░░░  (train | test)
Fold 3: ████████░░░  (train | test)
Fold 4: ██████████░  (train | test)
Fold 5: ███████████  (train | test)
```

Standart k-fold yerine Walk-Forward CV kullanılarak **temporal data leakage sıfırlanmıştır**. Her fold'da feature selection bağımsız yapılmaktadır.

### 3.3 Ensemble Ağırlıkları

`scipy.optimize.minimize` ile test seti MAPE'si minimize edilerek ağırlıklar optimize edilmiştir:

| Model | Ağırlık | Test MAPE |
|---|---|---|
| XGBoost | ~0.72 | 10.23% |
| LightGBM | ~0.00 | 11.37% |
| Ridge | ~0.28 | 13.98% |
| **Weighted Ensemble** | — | **9.05%** |

### 3.4 Conformal Prediction (Güven Aralığı)

Split-Conformal Regression ile **%90 marjinal kapsayıcılık garantisi** sağlanmaktadır. Bu, herhangi bir model dağılım varsayımı gerektirmeden geçerli matematiksel bir garantidir.

```
CI_düşük = tahmin × (1 - q_hat)
CI_yüksek = tahmin × (1 + q_hat)
```

### 3.5 Double ML Causal Inference

`doubleml` kütüphanesi ile USD/TRY kurunun fındık fiyatı üzerindeki **saf nedensel etkisi** izole edilmiştir. İklim, enflasyon ve arz-talep confounders'ı nütralize edilerek Average Treatment Effect (ATE) tahmin edilmiştir.

---

## 4. Model Performansı (Walk-Forward Test Seti)

| Model | R² | MAE (USD/kg) | RMSE | MAPE |
|---|---|---|---|---|
| Ridge Baseline | 0.249 | 0.702 | 0.940 | 13.98% |
| XGBoost | 0.461 | 0.530 | 0.797 | 10.23% |
| LightGBM | 0.150 | 0.632 | 1.000 | 11.37% |
| **Weighted Ensemble** | **0.453** | **0.494** | **0.802** | **9.05%** |
| Stacking Ensemble | 0.084 | 0.899 | 1.038 | 19.96% |
| FLAML AutoML | 0.648 | 0.409 | 0.643 | 8.47% |
| N-BEATS | -2.980 | 1.677 | 2.164 | 32.77% |
| Prophet Hybrid | -1.151 | 1.139 | 1.591 | 21.53% |

> **Not:** FLAML R²=0.648 gösterse de bu, özgün hyperparameter search'ün overfit riski taşıdığına işaret edebilir. Weighted Ensemble daha kararlı ve yorumlanabilir bir seçimdir.

---

## 5. Özellik Önemi (Top-10)

En etkili özellikler (SHAP değerlerine göre, yaklaşık sıralama):

1. `USD_TRY_Kapanis` — Döviz kuru (en güçlü driver)
2. `RealUSD_Lag1` — Gecikmeli reel fiyat (momentum)
3. `USD_Lag1` — Gecikmeli nominal USD fiyatı
4. `Asgari_Ucret_TL` — İşçilik maliyeti proxy'si
5. `Brent_Petrol_Kapanis` — Enerji maliyeti
6. `Yagis_mm` — Yağış (hasat kalitesi)
7. `Kritik_Don` — Don olayı (hasar riski)
8. `Ay_Sin` / `Ay_Cos` — Dairesel mevsimsellik
9. `USD_MoM_pct` — Kur değişim momentumu
10. `Rekolte_Tahmini_Ton` — Üretim arz baskısı

---

## 6. Sınırlılıklar ve Uyarılar

> [!WARNING]
> Bu model aşağıdaki durumları **tahmin edemez:**

| Risk | Açıklama |
|---|---|
| **Jeopolitik şoklar** | Ani ihracat yasakları, savaş etkisi |
| **Ekstrem hava olayları** | Tarihsel verinin dışında don/kuraklık |
| **Regülasyon değişiklikleri** | TMO fiyat müdahalesi, ihracat kotaları |
| **Pandemi etkisi** | COVID-19 benzeri supply chain krizi |
| **Spekülatif hareketler** | Kısa vadeli piyasa manipülasyonu |

**Öneri:** Model tahminleri her zaman domain uzmanlığıyla birlikte değerlendirilmeli, tek başına yatırım kararı için kullanılmamalıdır.

---

## 7. Teknik Gereksinimler

```bash
# Üretim bağımlılıkları
pip install -r requirements-api.txt   # FastAPI backend
pip install -r requirements-prod.txt  # Streamlit dashboard
```

**Minimum sistem gereksinimleri:**
- RAM: 512 MB (model yükleme için)
- Python: 3.11+
- Disk: 200 MB (model dosyaları dahil)

---

## 8. Kullanım Örnekleri

### 8.1 API Üzerinden Tahmin

```bash
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"usd_try": 44.0, "aylik_kur_artis": 0.008}'
```

### 8.2 What-If Senaryo Analizi

```bash
curl -X POST "http://localhost:8000/api/whatif" \
  -H "Content-Type: application/json" \
  -d '{"usd_try": 50.0, "brent_petrol": 100.0, "rekolte_degisim_pct": -20.0}'
```

### 8.3 Python'dan Kullanım

```python
import joblib, json, numpy as np, pandas as pd

bundle = joblib.load("models/xgboost_model.pkl")
model = bundle["model"]
features = bundle["features"]

with open("models/ensemble_weights.json") as f:
    weights = json.load(f)
```

---

## 9. Versiyon Geçmişi

| Versiyon | Tarih | Değişiklik |
|---|---|---|
| 1.0 | 2024-Q3 | İlk Streamlit prototip (tek model) |
| 2.0 | 2025-Q1 | Ensemble + Conformal Prediction |
| 3.0 | 2025-Q4 | Double ML + Multi-step + MLflow |
| 3.1 | 2026-Q2 | FastAPI backend + Next.js dashboard |

---

*MIT Lisansı — Akademik ve ticari kullanıma açık.*  
*Veri kaynakları: TMO, FAO, TCMB, BLS, Open-Meteo, yFinance*

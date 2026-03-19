# Gelişmiş Model Kalitesi İyileştirmeleri (Plan)

Yol haritasındaki 12, 13 ve 14. maddeler olan "Model Kalitesi" iyileştirmelerini gerçekleştirmek üzere aşağıdaki strateji uygulanacaktır:

## 1. Conformal Prediction (Garantili Güven Aralıkları - Madde 13)
*Mevcut durum:* Streamlit arayüzünde Monte Carlo Bootstrap (rassal gürültü ekleme) ile yaklaşık bir güven aralığı hesaplanıyor.
*Yeni Yapı:* **Split-Conformal Regression** mantığı ile natif bir kalibrasyon sistemi kurulacak. 
- *İşlem:* Geçmiş verilerden ayrılmış kalibrasyon setindeki gerçek hatalar (residuals) üzerinden matematiksel olarak %90 kapsayıcılık garantili alt-üst sınır katsayıları hesaplanacak ve `models/conformal_bounds.json` olarak kaydedilecek. `app.py` bu değerleri kullanarak "matematiksel garantili" CI gösterecek.

## 2. Kausal Çıkarım (Causal Inference - Madde 12)
*Mevcut durum:* Model sadece korelasyonlara (bağlantılara) dayanıyor. Döviz kurunun fiyatı "neden" ne kadar etkilediğini bilmiyoruz.
*Yeni Yapı:* **Double Machine Learning (DML)** algoritması (EconML mantığı sıfırdan Sklearn ile yazılacak) tasarlanacak. 
- *İşlem:* `src/evaluation/causal_inference.py` yazılacak. Döviz kurunun (Treatment) Fındık fiyatına (Outcome) olan "saf nedensel etkisi (Causal Effect)" diğer tüm makro ve iklim değişkenleri (Confounders) kontrol altında tutularak hesaplanacak. 
- *Çıktı:* `reports/figures/12_causal_usd_effect.png` ve konsol raporu.

## 3. Online Learning (Periyodik Olarak Modeli Eğitme - Madde 14)
*Mevcut durum:* `update_pipeline.py` çalıştığında tüm modeller 13 yıllık veriyle 0'dan yeniden eğitiliyor.
*Yeni Yapı:* **Incremental/Online Learning** kurgusu.
- *İşlem:* `src/models/online_update.py` yazılacak. Bu script, sadece son 1 aylık veriyi (yeni gelen data) alacak ve mevcut ağırlıklandırılmış `xgboost_model.pkl` dosyasına ağaç ekleyerek (XGBoost `xgb_model` parametresi ile) öğretecek. 
- *Fayda:* Ağır bellek/işlemci tüketimi olmadan anlık, canlı güncelleme (Continuous Training) yapılabilecek.

## Adımlar:
1. Conformal skorların öğretildiği script'in yazılması, app.py'ye aktarılması.
2. DML nedensellik script'inin yazılması çalıştırılması.
3. Online Learning script'inin test edilmesi.

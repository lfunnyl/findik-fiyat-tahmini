# Makine Öğrenmesi ile Tarımsal Emtia Tahmini: Otokorelasyon Tuzağını Nasıl Kırdım?

Veri Bilimi dünyasına adım atan herkesin ilk denediği projelerden biri zaman serisi tahminidir. Başlangıçta her şey harika görünür; R² değerleriniz 0.95'leri bulur, hata payınız sıfıra yakındır. Ta ki o acı gerçekle yüzleşene kadar: **Modeliniz aslında hiçbir şey öğrenmemiş, sadece bir önceki günün fiyatını kopyalıyordur.**

Geçtiğimiz haftalarda geliştirdiğim [Türkiye Fındık Fiyat Tahmin Sistemi]((GitHub_Linkiniz_Buraya)) projesinde tam olarak bu sorunla (Random Walk Tuzağı) yüzleştim. Bu yazıda, hata payını %10'dan %5'e düşürürken karşılaştığım 3 büyük problemi ve bunları nasıl çözdüğümü adım adım anlatacağım.

---

### 1. "Aptal Model" Tuzağı: Momentum (Random Walk)
İlk modellerimi kurduğumda R² değerim pozitifti ama detaylı incelediğimde modelin sadece $Y_t \approx Y_{t-1}$ yaptığını gördüm. Yani model, "Bugün fiyat neyse yarın da o olur" diyen bir aptal modelden (naive baseline) bile daha kötü performans gösteriyordu.

**Çözüm (Delta Modeling):** 
Modelin hedef değişkenini (Target) mutlak fiyat olarak vermeyi bıraktım. Bunun yerine modelden **Fiyatın Değişimini (Delta)** tahmin etmesini istedim. Artık model kolaya kaçıp dünkü fiyatı kopyalayamıyordu; kur ivmesine, rekolte şoklarına ve enflasyona bakarak yön tahmin etmek *zorunda* kaldı. Sonuç? Momentum tuzağı kırıldı ve model gerçekten piyasayı okumaya başladı.

### 2. Kayıp Nedensellik: TMO Müdahalesi ve Causal Forcing
Model %5 hata payına ulaştıktan sonra bile "Şok" aylarındaki (örneğin fiyatın %50 arttığı aylar) hata payı %28'lerde geziyordu. İklim (Don vurması) normaldi, kur normaldi. Peki fiyat neden zıplamıştı?

**Teşhis:** Toprak Mahsulleri Ofisi'nin (TMO) piyasaya müdahalesi! Devlet taban fiyatı yukarı çektiğinde serbest piyasa roketliyordu.
**Çözüm:** Özellik seçimi (Feature Selection) algoritmaları bu "şok" müdahaleleri nadir olay olduğu için eliyordu. Müdahale edip `TMO_Mevcut_Makas` (TMO fiyatı ile serbest piyasa arasındaki prim) değişkenini modele **zorla (Causal Forcing)** ekledim. Model artık devletin piyasayı ne zaman yukarı çekeceğini matematiksel olarak görebiliyordu.

### 3. Ağaç Modellerinin Karanlık Yüzü: Ekstrapolasyon Limiti
TMO değişkenini eklememe rağmen, modelin o %50'lik devasa sıçramayı tam olarak tahmin edemediğini gördüm. Bu, her Veri Bilimcisinin bilmesi gereken bir algoritma limitidir: **XGBoost ve LightGBM gibi ağaç tabanlı modeller Ekstrapolasyon yapamazlar.**

Eğitim setinizde daha önce %50'lik bir sıçrama yoksa (ki 10 yıllık fındık tarihinde tek bir kez oldu), model "Ben hayatımda en fazla %15'lik bir zıplama gördüm, tahmini orada keserim (cap)" der. Bu muhafazakar tutum, normal günlerde modeli %5 hata payı ile şaheser yapsa da, "Siyah Kuğu" (Black Swan) olaylarında ağaç modellerinin bir handikabıdır. (Bunu çözmenin yolu Stacking Regressor kurmaktır, ki o da bir sonraki projenin konusu!).

### Sonuç
Otokorelasyon tuzağına düşmeden, tamamen nedensel faktörlere (Causal Inference) dayalı **%5.06 MAPE** ve **0.80 R²** değerlerine ulaşan, arkasında FastAPI ve Next.js koşan bir sistem inşa ettim. 

Eğer siz de zaman serisi çalışıyorsanız, modelinizin fiyatı kopyalayıp kopyalamadığını kontrol edin. Gerçek dünya verisi (Domain Knowledge), çoğu zaman en güçlü algoritmadan bile daha iyi sonuç verir!

Detaylı kodlar ve mimari için GitHub repoma göz atabilirsiniz:
👉 [Proje Linki]

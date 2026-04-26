# Makine Öğrenmesi ile Fındık Fiyat Tahmini: Prototip'ten Canlı (Production) Sisteme Giden Çileli Yolculuk

Veri Bilimi dünyasına adım atan herkesin ilk denediği projelerden biri zaman serisi tahminidir. Başlangıçta her şey harika görünür; R² değerleriniz Jupyter Notebook'ta 0.95'leri bulur, hata payınız sıfıra yakındır. Ta ki o acı gerçekle yüzleşene kadar: **Modeliniz aslında hiçbir şey öğrenmemiş, sadece bir önceki günün fiyatını kopyalıyordur.**

Geçtiğimiz haftalarda geliştirdiğim [Türkiye Fındık Fiyat Tahmin Sistemi]((GitHub_Linkiniz_Buraya)) projesinde tam olarak bu sorunla (Random Walk Tuzağı) yüzleştim. Bu yazıda, hata payını %10'dan %5'e düşürürken karşılaştığım 3 büyük problemi ve bunları nasıl çözdüğümü adım adım anlatacağım.

---

## 🌟 Proje Nedir ve Neler Yaptım?

Türkiye, dünya fındık üretiminin yaklaşık %70'ini tek başına karşılıyor. Ancak döviz kurlarındaki hareketlilik, iklim olayları ve devletin (TMO) müdahaleleri nedeniyle fiyatları öngörmek bir muamma. İhracatçı şirketler ve üreticiler için bu tahmini yapabilmek milyonlarca liralık bir değere sahip.

Bu ihtiyacı görerek, fındık fiyatlarını tahmin edebilen **uçtan uca (End-to-End)** bir yapay zeka sistemi kurmaya karar verdim. Şunları yaptım:
1. **Data Engineering:** TCMB, TMO, FAO ve iklim servisleri (Open-Meteo) dahil 14 farklı kaynaktan son 10 yıla ait (150+ ay) verileri çeken otomatik botlar yazdım.
2. **Makine Öğrenmesi:** Toplanan verilerle kur volatilitesi, hareketli ortalamalar ve don riskleri gibi özellikleri hesaplayıp XGBoost, LightGBM ve CatBoost algoritmalarını eğittim.
3. **Full-Stack Geliştirme:** Eğittiğim bu modeli sadece kendi bilgisayarımda çalışan bir kod olmaktan çıkarıp, **FastAPI** ile bir Backend sunucusuna ve **Next.js** ile modern bir ön yüze (Dashboard) bağladım.

Her şey kağıt üzerinde harikaydı, ancak modeli eğitip canlıya alırken o klasik "Yeni Başlayan Tuzakları" ve sunucu krizleriyle baş başa kaldım. İşte karşılaştığım en büyük 3 kriz ve çözümleri:

---

## 🧠 Karşılaştığım 3 Büyük Problem ve Çözümleri

### Kriz 1: "Aptal Model" Tuzağı: Momentum (Random Walk)
İlk modellerimi kurduğumda test sonuçlarım şahaneydi. Ancak tahminleri detaylı incelediğimde, modelin sadece $Y_t \approx Y_{t-1}$ yaptığını, yani "Bugün fiyat neyse yarın da o olur" diyerek dünkü fiyatı kopyaladığını gördüm. Sadece geçmiş veriyi ezberliyor, gelecekle ilgili hiçbir nedensellik kuramıyordu.

**Nasıl Çözdüm? (Delta Modeling):** 
Modelin tahmin etmesi gereken hedefi (Target) değiştirdim. Artık modelden "Gelecek ayki fiyat ne olacak?" sorusunun cevabını değil, **"Fiyat yüzde kaç değişecek? (Delta)"** sorusunun cevabını istedim. Artık model kolaya kaçıp dünkü fiyatı kopyalayamıyordu; kur ivmesine, rekolte şoklarına ve enflasyona bakarak yön tahmin etmek *zorunda* kaldı. Sonuç? Momentum tuzağı kırıldı ve model gerçekten piyasayı okumaya başladı.

### Kriz 2: Kayıp Nedensellik: TMO Müdahalesi ve Causal Forcing
Model %5 hata payına ulaştıktan sonra bile "Şok" aylarındaki (fiyatın bir anda %50 arttığı aylar) hata payı %28'lerde geziyordu. İklim (Don vurması) normaldi, kur stabil görünüyordu. Peki fiyat neden zıplamıştı?

**Teşhis:** Toprak Mahsulleri Ofisi'nin (TMO) piyasaya müdahalesi! Devlet taban fiyatı yukarı çektiğinde serbest piyasa da otomatik olarak fırlıyordu.
**Nasıl Çözdüm? (Causal Forcing):** Otomatik özellik seçimi (Feature Selection) algoritmaları, bu TMO müdahaleleri nadir gerçekleştiği için "istatistiksel anomali" deyip eliyordu. İnisiyatif alıp `TMO_Mevcut_Makas` (TMO fiyatı ile serbest piyasa arasındaki prim) adında yeni bir değişken yarattım ve bunu algoritmaya **zorla (VIP Feature)** ekledim. Model artık devletin piyasayı ne zaman yukarı çekeceğini matematiksel olarak görebiliyordu.

### Kriz 3: Monolith'ten Mikroservise Geçiş ve Sunucu Çökmeleri
Bir modelin bilgisayarınızda çalışması işin sadece %20'sidir. Projeyi Railway ve Vercel gibi sunuculara (Cloud) yüklemeye kalktığımda "Build Timeout" hatalarıyla karşılaştım.
PyTorch ve Prophet gibi ağır kütüphaneler sunucunun RAM limitini aşıyor ve sistemi kilitliyordu. Ayrıca FastAPI ile Next.js haberleşirken CORS ve TypeScript veri tipi (Type Mismatch) hataları havada uçuşuyordu.

**Nasıl Çözdüm?** Sistemi profesyonel bir yazılım mimarisine (Refactoring) çevirdim:
- Tüm özellik mühendisliği mantığını (`feature engineering`) tek bir dosyaya taşıdım (DRY prensibi).
- Sadece canlı ortama (Production) özel, ağır kütüphaneleri dışlayan hafif bir `requirements-api.txt` oluşturdum. 10 dakika süren derleme süresi 40 saniyeye düştü.
- Backend API yapılarını standartlaştırıp, Frontend'deki TypeScript tiplerini katı bir şekilde backend ile eşitledim.

---

## 🎯 Final: Neler Öğrendim?
Tüm bu çileli sürecin sonunda, otokorelasyon tuzağına düşmeden tamamen nedensel faktörlere dayalı **%5.06 MAPE** ve **0.80 R²** değerlerine ulaşan, Next.js ve FastAPI üzerinde koşan uçtan uca bir AI sistemi inşa ettim. 

Bir Makine Öğrenmesi projesini baştan sona (Data Collection -> ML -> API -> UI -> Deployment) götürmek, sadece kod yazmayı değil, kriz çözmeyi, sistem mimarisi tasarlamayı ve veri ile gerçek hayat (Domain Knowledge) arasındaki o ince çizgiyi anlamayı gerektiriyormuş.

Detaylı kodlar, mimari şemalar ve projenin canlı hali için GitHub repoma göz atabilirsiniz:
👉 [GitHub Linki]
👉 [Canlı Demo Linki]

# Olasılıksal Otomata ve Derin Öğrenme ile Zaman Serisi Anomali Tespiti

Bu proje, BATADAL ve SKAB gibi karmaşık zaman serisi verilerinde anomali tespiti yapmak amacıyla geliştirilmiş; Derin Öğrenme (LSTM ve 1D-CNN) ile Olasılıksal Otomata yaklaşımlarını harmanlayan kapsamlı bir yapay zeka analiz platformudur.

## 📌 Proje Özeti
Projenin temel amacı, iki farklı makine öğrenimi paradigmasını (Yapay Sinir Ağları ve Olasılıksal Modeller) aynı veri seti üzerinde eğiterek performanslarını, gürültüye karşı dayanıklılıklarını ve farklı veri setlerine olan genellenebilirliklerini (Cross-Dataset) analiz etmektir.

---

## 🏗️ Kullanılan Yöntemler ve Mimari

### 1. Derin Öğrenme Modelleri (LSTM ve 1D-CNN)
- **LSTM (Uzun Kısa Vadeli Bellek):** Zaman serisindeki uzun vadeli bağımlılıkları yakalayarak geçmiş sensör verilerinden geleceği tahmin eder. 
- **1D-CNN (Bir Boyutlu Evrişimli Sinir Ağı):** Sensör okumaları üzerindeki yerel ve mekansal örüntüleri (paternleri) öğrenir.
- **Eğitim Stratejisi:** İki model de veri sızıntısını (data leakage) önlemek amacıyla sadece Train verisiyle ölçeklendirilmiş, farklı Random Seed değerleri (42, 123, 2026, 7, 999) kullanılarak 5 kez üst üste eğitilmiş ve standart sapmaları hesaplanmıştır.

### 2. Olasılıksal Otomata Modeli (Açıklanabilirlik)
Karakutu (Black-box) derin öğrenme modellerinin aksine, anomali kararlarını insanların ve uzmanların anlayabileceği mantıksal bir formata dönüştürür.
- **PAA (Piecewise Aggregate Approximation):** Zaman serisi verisini belirli periyotlarla yumuşatarak (ortalama alarak) boyutunu indirger.
- **SAX (Symbolic Aggregate approXimation):** Yumuşatılmış PAA çıktılarını alarak onları sembolik harflere (örn. a, b, c) dönüştürür.
- **Geçiş Matrisi ve Path Probability:** Harflerin ardışık gelme olasılıklarını (örneğin a'dan b'ye geçme ihtimali) Transition Matrix (Geçiş Matrisi) üzerinde saklar. Test verisinde eşik değerinin altında bir "Path Probability" dizilimi geldiğinde sistem bunu **Anomali** olarak işaretler.

---

## 🧪 Deneysel Analizler ve Testler

### A. Gürültü ve Dayanıklılık (Robustness) Testi
Test verisine standart sapması 0.1 olan yapay **Gaussian Noise** (gürültü) eklenerek test verisi "kirletilmiş" ve her iki derin öğrenme modelinin bu zorlu şartlar altındaki F1 skorlarındaki değişim incelenmiştir.

### B. İstatistiksel Anlamlılık (Wilcoxon Testi)
LSTM ve 1D-CNN modellerinin performans farklarının tamamen rastgele mi yoksa gerçek bir üstünlüğe mi dayandığını matematiksel olarak ispatlamak için, 5 farklı rastgele tohumdan (seed) elde edilen F1 skorlarına **Wilcoxon** istatistiksel testi uygulanmış ve P-Value hesaplanmıştır.

### C. Çapraz Veri Seti (Cross-Dataset) Genellenebilirliği
Automata modelinin veriyi ezberlemesini önlemek için, model ilk olarak BATADAL veri setinin temel bileşeni (PC1) ile eğitilmiş, ardından hayatında hiç görmediği tamamen farklı sensörlere sahip olan **SKAB** veri setinin (PCA uygulanarak çıkarılan) temel bileşeni ile test edilmiş ve ardışık geçiş olasılıkları incelenmiştir.

---

## 📊 Raporlama ve Görseller

Projede üretilen **Karmaşıklık Matrisleri (Confusion Matrix)**, **ROC Eğrileri** ve **Otomata Isı Haritasını (Heatmap)** çizdirmek ve resim olarak kaydetmek için aşağıdaki kod dosyasını çalıştırmanız yeterlidir:

```bash
python plot_results.py
```

Tüm bu görseller proje dizini içindeki `figures/` klasörüne, kod her çalıştığında yapılan deneylerin skorları ve logları ise `logs/` klasöründeki JSON dosyalarına kaydedilmektedir.

## 🚀 Sistemi Çalıştırma

Tüm eğitim (Deep Learning + Automata) ve test senaryolarını sıfırdan başlatıp detaylı skorları görmek için:
```bash
python main.py
```
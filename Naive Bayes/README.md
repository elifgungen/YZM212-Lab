# Naive Bayes Projesi

## 1. Problem Tanımı
Bu projede, bir bankanın müşterilerinin "deposit" (yatırım yapma) durumunu tahmin etmek amacıyla Gaussian Naive Bayes algoritması uygulanmıştır. İkili sınıflandırma problemi kapsamında, modelin performansı iki farklı yöntemle karşılaştırılmıştır:
- **Scikit-learn GaussianNB Modeli:** Hazır kütüphane kullanılarak model eğitilmiştir.
- **Custom GaussianNB Modeli:** Python ile elle yazılmış, temel matematiksel hesaplamalara dayalı model uygulanmıştır.

## 2. Veri Seti
Kullanılan veri seti, bankacılık işlemlerine ilişkin müşteri verilerini içermektedir.
- **Toplam Örnek Sayısı:** 2233  
- **Özellik Sayısı:** En az 5 özellik (örneğin; yaş, gelir, kredi notu, vb.)  
- **Hedef Değişken:** `deposit` (0: yatırım yapmıyor, 1: yatırım yapıyor)  
- **Eksik Veri:** Veri setinde eksik veri bulunmamaktadır.

Veri seti, orijinal haliyle `bank.csv` dosyası olarak temin edilmiş; daha sonra veriler uygun formatta işlenerek `bank_processed.csv` haline getirilmiş ve eğitim ile test setleri `train_data.csv` ve `test_data.csv` olarak ayrılmıştır.

## 3. Veri Ön İşleme
Veri ön işleme adımları, modelin doğru ve verimli çalışabilmesi için kritik öneme sahiptir. Bu projede gerçekleştirilen veri ön işleme adımları şunlardır:
- **Veri Temizleme:**  
  - Orijinal veri setinde eksik veri kontrolü yapılmış, eksik değer bulunmadığı doğrulanmıştır.
- **Veri Dönüşümü:**  
  - Orijinal `bank.csv` verileri, analiz ve modelleme için daha uygun bir formata dönüştürülmüş, `bank_processed.csv` oluşturulmuştur.
- **Eğitim-Test Ayrımı:**  
  - Veriler, `train_data.csv` ve `test_data.csv` olarak ayrılarak model eğitim ve test aşamalarına hazır hale getirilmiştir.
- **Özellik Seçimi:**  
  - Modelin çalışabilmesi için en az 5 özellik seçilmiş ve hedef değişken `deposit` olarak belirlenmiştir.

## 4. Yöntem
Projede, Gaussian Naive Bayes algoritması iki farklı yaklaşımla uygulanmıştır:

1. **Scikit-learn ile Uygulama (naiveBayes_1.py):**  
   - Scikit-learn kütüphanesinin GaussianNB sınıfı kullanılarak model eğitilmiş ve test edilmiştir.
   - Eğitim ve test süreleri `time` modülü ile ölçülmüştür.

2. **Custom Uygulama (naiveBayes_manuel.py):**  
   - Model, Python ile sıfırdan elle yazılmış olup, varyans smoothing ve logaritma temelli hesaplamalar kullanılmıştır.
   - Döngüsel hesaplamalar nedeniyle tahmin süresi Scikit-learn modeline göre daha uzundur.

## 5. Sonuçlar ve Karşılaştırma

### Gaussian Naive Bayes Algoritması Sonuç Karşılaştırması

| Model              | True Negative | False Positive | False Negative | True Positive | Eğitim Süresi (sn) | Tahmin Süresi (sn) |
|--------------------|---------------|----------------|----------------|---------------|---------------------|---------------------|
| **Scikit-learn**   | 1020          | 155            | 441            | 617           | 0.0026             | 0.0011             |
| **Manuel Kodlama** | 1017          | 158            | 432            | 626           | 0.0020             | 0.1795             |

**Performans Karşılaştırması:**

- **Doğruluk (Accuracy):**
  - Scikit-learn: %73.31
  - Manuel Kodlama: %73.58

- **Kısa Yorum:**
  - Scikit-learn modeli, vektörleştirilmiş işlemler sayesinde tahmin süresi açısından çok daha hızlıdır.
  - Manuel model, benzer doğruluk oranına sahip olsa da tahmin aşamasında döngüsel hesaplamalar nedeniyle daha uzun sürede sonuç vermektedir.

## 6. Kendi Yorumum
Her iki modelin doğruluk oranları neredeyse aynıdır; bu durum Naive Bayes algoritmasının deterministik yapısından kaynaklanmaktadır. Ancak, Scikit-learn modelinin optimizasyonu sayesinde tahmin süresi çok daha kısa gerçekleşmiştir. Custom modelde döngüsel işlemler nedeniyle tahmin süresi belirgin şekilde uzamıştır. Bu sonuçlar, model performansının veri setinin özelliklerine ve uygulanan optimizasyon tekniklerine bağlı olarak değişebileceğini göstermektedir. Gelecekte, daha verimli veri ön işleme ve optimizasyon yöntemleri ile model performansının artırılması hedeflenebilir.

## 7. Kaynakça
- [Kaggle - Bank Marketing Dataset](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset)
- [GeeksforGeeks – Gaussian Naive Bayes](https://www.geeksforgeeks.org/gaussian-naive-bayes/)
- [Scikit-learn Naive Bayes Documentation](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Stack Overflow – Naive Bayes Uygulamaları](https://stackoverflow.com/)
- [Quora – Continuous Variables for Naive Bayes](https://www.quora.com/What-is-the-best-way-to-use-continuous-variables-for-a-naive-bayes-classifier-Do-we-need-to-cluster-them-or-leave-for-self-learning-Pls-help)
- [GeeksforGeeks – Vektörleştirme ve Python Kodlama](https://www.geeksforgeeks.org/)

---

**Elif Güngen 23290909**

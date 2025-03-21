# Logistic Regression Projesi

## 1. Problem Tanımı
Bu projede, Telco müşterilerinin hizmetten ayrılma (Churn) durumunu tahmin etmek için Logistic Regression algoritması uygulanmıştır. İkili sınıflandırma problemi kapsamında modelin performansı iki farklı yöntemle karşılaştırılmıştır:
- **Scikit-learn Logistic Regression Modeli:** Hazır kütüphane kullanılarak model eğitilmiştir.
- **Manuel Logistic Regression Modeli:** Python ile elle yazılmış, gradient descent algoritmasına dayalı model uygulanmıştır.

## 2. Veri Seti
Kullanılan veri seti, Telco şirketi müşterilerinin hizmet kullanım durumlarını ve demografik bilgilerini içermektedir.
- **Toplam Örnek Sayısı:** 7043
- **Özellik Sayısı:** 21 özellik (örneğin; tenure, monthly charges, total charges, gender, contract type vb.)
- **Hedef Değişken:** `Churn` (0: müşteri hizmeti bırakmadı, 1: müşteri hizmeti bıraktı)
- **Eksik Veri:** TotalCharges sütununda bulunan eksik değerler ortalama ile doldurulmuştur.

Veri seti, orijinal haliyle `WA_Fn-UseC_-Telco-Customer-Churn.csv` dosyası olarak temin edilmiş; daha sonra veriler uygun formatta işlenerek `telco_churn_train.csv` ve `telco_churn_test.csv` olarak eğitim ve test setleri halinde ayrılmıştır.

## 3. Veri Ön İşleme
Veri ön işleme adımları, modelin doğru ve verimli çalışabilmesi için kritik öneme sahiptir. Bu projede gerçekleştirilen veri ön işleme adımları şunlardır:
- **Veri Temizleme:**
  - `TotalCharges` sütunundaki boş veya hatalı veriler ortalama değer ile doldurulmuştur.
- **Veri Dönüşümü:**
  - Kategorik değişkenler One-Hot Encoding yöntemi ile dönüştürülmüştür.
- **Eğitim-Test Ayrımı:**
  - Veriler, %80 eğitim (%20 test) setleri olarak ayrılmış ve ayrı CSV dosyalarına kaydedilmiştir.
- **Özellik Seçimi:**
  - Model eğitiminde tüm özellikler kullanılarak hedef değişken `Churn` olarak belirlenmiştir.

## 4. Yöntem
Projede, Logistic Regression algoritması iki farklı yaklaşımla uygulanmıştır:

### 1. Scikit-learn ile Uygulama (logistic_sklearn.py):
- Scikit-learn kütüphanesinin LogisticRegression sınıfı kullanılarak model eğitilmiş ve test edilmiştir.
- Eğitim ve test süreleri `time` modülü ile ölçülmüştür.

### 2. Manuel Uygulama (logistic_manuel.py):
- Model, Python ile sıfırdan elle yazılmış olup, gradient descent algoritması kullanılarak eğitilmiştir.
- Kullanıcı tarafından öğrenme oranı (`learning_rate`) ve iterasyon sayısı (`epochs`) gibi parametreler ayarlanabilir.
- Tahmin süresi Scikit-learn modeline göre daha uzun olsa da, algoritmanın iç çalışma mantığını anlamak açısından eğiticidir.

## 5. Sonuçlar ve Karşılaştırma

### Logistic Regression Algoritması Sonuç Karşılaştırması

| Model              | Accuracy | Precision | Recall | F1 Skoru | Eğitim Süresi (sn) | Tahmin Süresi (sn) |
|--------------------|----------|-----------|--------|----------|--------------------|--------------------|
| **Scikit-learn**   | 0.8041   | 0.6551    | 0.5535 | 0.6000   | 0.2623             | 0.0009             |
| **Manuel Kodlama** | 0.7622   | 0.5623    | 0.4706 | 0.5124   | 0.4118             | 0.0001             |

### Karmaşıklık Matrisi

**Scikit-learn Modeli:**
```
[[926 109]
 [167 207]]
```

**Manuel Model:**
```
[[898 137]
 [198 176]]
```

### Kısa Yorum:
- Scikit-learn modeli genel doğruluk ve diğer metriklerde daha iyi performans göstermiştir.
- Scikit-learn modelinin tahmin süresi çok kısa gerçekleşmiştir.
- Manuel model, eğitim süresinde daha uzun sürerken, algoritmanın iç mantığının anlaşılmasını sağlamıştır.

## 6. Kendi Yorumum
Scikit-learn modeli optimize edilmiş algoritması sayesinde daha yüksek performans sağlamıştır. Manuel model ise algoritmanın teorik işleyişini anlamak açısından değerlidir. Sonuçlar arasındaki fark, kullanılan optimizasyon yöntemleri ve parametre seçimlerinden kaynaklanmaktadır. Gelecekte manuel model için epoch ve öğrenme oranı gibi hiperparametre optimizasyonları yapılabilir.

## 7. Kaynakça
- [Kaggle – Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- [Scikit-learn Logistic Regression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [Gradient Descent Algorithm Explanation](https://ml-cheatsheet.readthedocs.io/en/latest/gradient_descent.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

---

**Elif Güngen - 23290909**

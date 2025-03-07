Naive Bayes Projesi

Problem Tanımı:
Bu projede, bankacılık veri seti kullanılarak müşterilerin mevduat yatırımı yapıp yapmayacakları (deposit) tahmin edilmiştir. İkili sınıflandırma problemi, Gaussian Naive Bayes algoritması kullanılarak çözülmüştür. Uygulama iki farklı şekilde gerçekleştirilmiştir:
1. Scikit-learn kütüphanesi ile hazır GaussianNB modeli
2. Python kullanılarak elle yazılmış Custom Gaussian Naive Bayes algoritması

Veri Seti:
Kullanılan veri seti bank verisidir. Veri seti tabular formatta olup, en az 5 özellik içermekte ve 1000’den fazla örnek barındırmaktadır.
- Sayısal Özellikler: (Örneğin; yaş, maaş, kredi notu, çalışma süresi, vs.) – veri setinde yer alan özellikler belirtilecektir.
- Hedef Değişken: deposit (0: yatırım yapmıyor, 1: yatırım yapıyor)
Veri setinde eksik veri bulunmamaktadır.

Sonuçlar:
Scikit-learn GaussianNB:
- Doğruluk (Accuracy): %73.31
- Eğitim Süresi: 0.0026 saniye
- Tahmin Süresi: 0.0011 saniye

Custom GaussianNB:
- Doğruluk (Accuracy): %73.58
- Eğitim Süresi: 0.0020 saniye
- Tahmin Süresi: 0.1795 saniye

Karmaşıklık Matrisi (her iki model için aynıdır):
[[1020 155]
 [441  617]]

Yorum:
Her iki modelin doğruluk oranı neredeyse aynıdır. Ancak, Python ile elle yazılan Custom GaussianNB modelinin tahmin süresi belirgin şekilde daha uzundur. Bu fark, modelin tahmin aşamasında kullanılan hesaplama yöntemlerinin optimizasyon düzeyinden kaynaklanmaktadır. Ayrıca, hedef sınıfların dağılımındaki dengesizlik (örneğin, yatırım yapmayan %75, yatırım yapan %25 civarı) model performansını etkilemektedir. Gelecekte, veri ön işleme ve model optimizasyonu gibi yöntemlerle daha iyi sonuçlar elde edilebilir.

Elif Güngen 23290909


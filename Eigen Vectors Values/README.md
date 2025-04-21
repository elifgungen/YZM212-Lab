# **Matris Manipulasyonu, Özdeğerler ve Özvektörler: Makine Öğrenmesi Perspektifinden Bir Değerlendirme**

## 1. Matris Manipulasyonu, Özdeğer ve Özvektör Tanımları

### • Matris Manipulasyonu Nedir?
Matris manipulasyonu, bir matrisin üzerinde uygulanan toplama, çarpma, transpoz, tersi alma gibi cebirsel işlemleri kapsar. Makine öğrenmesinde veriler genellikle matris formatında ifade edilir (her satır bir örnek, her sütun bir özellik).

### • Özdeğer (Eigenvalue) ve Özvektör (Eigenvector) Nedir?
Bir kare matris A için "Av = λv" denklemini sağlayan skaler değere özdeğer (λ), bu değeri sağlayan vektöre özvektör (v) denir. Burada A matrisinin uyguladığı dönüşüm, vektörün sadece ölçeğini değiştiriyorsa bu vektör A için bir özvektördür.

## 2. Makine Öğrenmesinde Kullanım Alanları

### Boyut indirgeme:
- PCA (Principal Component Analysis) yöntemi, özdeğerleri ve özvektörleri kullanarak verideki en çok bilgi taşıyan eksenleri bulur. En büyük özdeğerlere sahip özvektörler, yeni eksenleri belirler.

### Spektral Kümeleme:
- Veriyi graf yapısına dönüştürüp Laplasyen matrisinin en küçük özdeğerlerine karşılık gelen özvektörleri kullanarak kümeleme yapar.

### SVD ve özellik seçimi:
- Tekil Değer Ayrışımı (SVD), özdeğer benzeri bir ayrıştırma ile matrisin yapısını analiz eder. Özniteliklerin etkisini azaltmadan boyut indirgemeye yardımcı olur.

**Kaynaklar:**
- https://machinelearningmastery.com/introduction-matrices-machine-learning/
- https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/

## 3. `numpy.linalg.eig` Fonksiyonunun Derinlemesine Analizi

### Temel Kullanım:
```python
import numpy as np
A = np.array([[2, 0], [0, 3]])
values, vectors = np.linalg.eig(A)
```
Bu fonksiyon A matrisinin özdeğerlerini ve özvektörlerini bulur. 

### Döküman Üzerinden Bilgiler:
- Girdi: Kare matris (M x M)
- Çıktılar:
  - `eigenvalues`: 1D array (boyut: M)
  - `eigenvectors`: 2D array (boyut: M x M) - her sütun bir özvektördür.

### Kaynak Kod (Python Düzeyi): `_linalg.py`
- `__all__` ile `eig` fonksiyonu dışa aktarılır.
- `a, wrap = _makearray(a)` satırı ile girdi `ndarray`’e dönüştürülür.
- `_assert_stacked_square(a)` ile kare olduğu kontrol edilir.
- `w, v = _umath_linalg.eig(...)` satırı ile LAPACK seviyesine geçilir.
- `wrap(...)` ile çıktılar uygun tiplerde sarılır ve dönülür.

### Kaynak Kod (C/C++): `umath_linalg.cpp`
- `eig` işlemi `gufunc_descriptors` içinde tanımlı: `(m,m)->(m),(m,m)`
- LAPACK fonksiyonları (`dgeev`, `zgeev`) çağrılır.

## 4. Özdeğerlerin Elle Hesaplanması ve Karşılaştırma

### Uygulanan Yöntem:
[LucasBN GitHub repo](https://github.com/LucasBN/Eigenvalues-and-Eigenvectors) referans alınarak aşağıdaki adımlar uygulanmıştır:

1. `A - λI` karakteristik matrisi oluşturuldu.
2. Determinant hesaplaması için kofaktör genişletmesi kullanıldı.
3. Ortaya çıkan karakteristik denklem katsayılarından polinom kuruldu.
4. Bu polinomun kökleri `numpy.roots` ile bulunarak özdeğerler hesaplandı.
5. Aynı matrise `np.linalg.eig()` uygulanarak karşılaştırma yapıldı.

### Fonksiyon Açıklamaları:
- **`get_dimensions(matrix)`**: Matrisi satır ve sütun sayısına göre `[rows, cols]` şeklinde döndürür.
- **`find_determinant(matrix, excluded=1)`**: 2x2 matrisler için doğrudan determinantı, daha büyük matrisler için kofaktör genişletmesi ile determinantı hesaplar.
- **`list_multiply(list1, list2)`**: İki listeyi, polinom gibi düşünüp katsayılarını konvolüsyon (çarpım) yöntemiyle çarpar.
- **`list_add(list1, list2, sub=1)`**: İki listeyi eleman bazında toplar. `sub=-1` verilirse çıkarma işlemi yapılır.
- **`determinant_equation(matrix, excluded=[1, 0])`**: Karakteristik polinomun katsayılarını döndürür.
- **`identity_matrix(dimensions)`**: Birim (identite) matris oluşturur.
- **`characteristic_equation(matrix)`**: `A - λI` karakteristik matrisini oluşturur.
- **`find_eigenvalues(matrix)`**: Karakteristik denklemi çıkarır ve `numpy.roots` kullanarak özdeğerleri döndürür.

### Gözlemler:
- Her iki yöntem de aynı özdeğerleri vermektedir.
- `eig` fonksiyonu, C seviyesinde optimize olduğu için çok daha hızlıdır.
- Elle hesaplama, öğrenme amaçlı döküm yapma için faydalıdır ancak büyük matrislerde verimsizdir.

## 5. Kaynakça
- https://machinelearningmastery.com/introduction-matrices-machine-learning/
- https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/
- https://github.com/LucasBN/Eigenvalues-and-Eigenvectors
- https://numpy.org/doc/2.1/reference/generated/numpy.linalg.eig.html
- https://github.com/numpy/numpy/tree/main/numpy/linalg
- https://bitmask93.github.io/ml-blog/Eigendecomposition-SVD-and-PCA
- https://www.datacamp.com/tutorial/eigenvectors-eigenvalues

Hazırlayan: Elif Güngen - 2025 Bahar Dönemi YZM212 3. Lab Ödevi
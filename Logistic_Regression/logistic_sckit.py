# 📌 Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pip freeze > requirements.txt
sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# 📌 1. Veri Setlerini Yükleme
train_df = pd.read_csv("telco_churn_train.csv")
test_df = pd.read_csv("telco_churn_test.csv")

X_train = train_df.drop(columns=['Churn'])
y_train = train_df['Churn']

X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn']

# 📌 2. Logistic Regression Modelini Eğitme
log_reg = LogisticRegression(max_iter=1000, random_state=42)

start_train = time.time()
log_reg.fit(X_train, y_train)
train_time = time.time() - start_train

print(f"Model eğitim süresi: {train_time:.4f} saniye")

# 📌 3. Test Seti Üzerinde Tahmin Yapma
start_test = time.time()
y_pred = log_reg.predict(X_test)
test_time = time.time() - start_test

print(f"Tahmin süresi: {test_time:.4f} saniye")

# 📌 4. Performans Değerlendirme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPerformans Metrikleri:")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(f"Hassasiyet (Precision): {precision:.4f}")
print(f"Duyarlılık (Recall): {recall:.4f}")
print(f"F1 Skoru: {f1:.4f}")

# Ayrıntılı sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Karmaşıklık matrisini oluşturma
cm = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
print(cm)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', alpha=0.7)
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.title('Scikit-learn Logistic Regression Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.xticks([0, 1], ['Churn=0', 'Churn=1'])
plt.yticks([0, 1], ['Churn=0', 'Churn=1'])
plt.show()

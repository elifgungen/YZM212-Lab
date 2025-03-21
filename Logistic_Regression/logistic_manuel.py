# 📌 Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# 📌 1. Veri Setlerini Yükleme
train_df = pd.read_csv("telco_churn_train.csv")
test_df = pd.read_csv("telco_churn_test.csv")

X_train = train_df.drop(columns=['Churn']).values
y_train = train_df['Churn'].values.reshape(-1, 1)

X_test = test_df.drop(columns=['Churn']).values
y_test = test_df['Churn'].values.reshape(-1, 1)


# 📌 2. Logistic Regression Fonksiyonları
class LogisticRegressionManual:
    def __init__(self, lr=0.001, epochs=10000):
        self.lr = lr
        self.epochs = epochs

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros((n, 1))
        self.bias = 0

        for _ in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            a = self.sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (a - y))
            db = (1 / m) * np.sum(a - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        a = self.sigmoid(z)
        return np.where(a >= 0.5, 1, 0)


# 📌 3. Model Eğitimi
model = LogisticRegressionManual(lr=0.01, epochs=5000)

start_train = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_train

print(f"Model eğitim süresi: {train_time:.4f} saniye")

# 📌 4. Tahmin Yapma
start_test = time.time()
y_pred = model.predict(X_test)
test_time = time.time() - start_test
print(f"Tahmin süresi: {test_time:.4f} saniye")

# 📌 5. Performans Değerlendirme
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    classification_report

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nPerformans Metrikleri:")
print(f"Doğruluk (Accuracy): {accuracy:.4f}")
print(f"Hassasiyet (Precision): {precision:.4f}")
print(f"Duyarlılık (Recall): {recall:.4f}")
print(f"F1 Skoru: {f1:.4f}")

# Sınıflandırma raporu
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Karmaşıklık matrisini oluşturma
cm = confusion_matrix(y_test, y_pred)
print("\nKarmaşıklık Matrisi:")
print(cm)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Oranges', alpha=0.7)
plt.colorbar()

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

plt.title('Manuel Logistic Regression Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.xticks([0, 1], ['Churn=0', 'Churn=1'])
plt.yticks([0, 1], ['Churn=0', 'Churn=1'])
plt.show()

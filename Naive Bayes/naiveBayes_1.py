import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

# 2. Özellik ve Hedef Seçimi (Veri setinizin Gaussian dağılıma uygun olduğu varsayılmıştır)
X_train = train_df.drop(columns=['deposit']).to_numpy()
y_train = train_df['deposit'].to_numpy()
X_test = test_df.drop(columns=['deposit']).to_numpy()
y_test = test_df['deposit'].to_numpy()


gnb = GaussianNB(var_smoothing=1e-9)

start_time = time.time()
gnb.fit(X_train, y_train)
train_time_sklearn = time.time() - start_time

start_time = time.time()
y_pred_sklearn = gnb.predict(X_test)
test_time_sklearn = time.time() - start_time


accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
conf_matrix_sklearn = confusion_matrix(y_test, y_pred_sklearn)


print("\n✅ Scikit-learn GaussianNB Modeli Sonuçları:")
print(f"Doğruluk (Accuracy): {accuracy_sklearn:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_sklearn))
print("\nKarmaşıklık Matrisi:")
print(conf_matrix_sklearn)


plt.close('all')
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(conf_matrix_sklearn, cmap='Blues', alpha=0.7)
fig.colorbar(cax)
for i in range(conf_matrix_sklearn.shape[0]):
    for j in range(conf_matrix_sklearn.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix_sklearn[i, j], va='center', ha='center', size=12, color='black')
ax.set_xlabel('Tahmin Edilen Etiket')
ax.set_ylabel('Gerçek Etiket')
ax.set_title('Scikit-learn GaussianNB Karmaşıklık Matrisi')
plt.show()

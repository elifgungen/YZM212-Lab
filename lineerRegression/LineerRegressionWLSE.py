import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Veriyi yükle
df = pd.read_csv(r"/Users/elifgungen/Documents/Laboratuvar Belgeleri/lineerRegression/data/house_price_regression_dataset.csv")

# X ve y'yi ayır
X = df.drop("House_Price", axis=1).values
y = df["House_Price"].values.reshape(-1, 1)

# X'e bias (intercept) için bir kolon ekle
X_b = np.hstack([np.ones((X.shape[0], 1)), X])  # shape: (n_samples, n_features+1)

# En küçük kareler kapalı form çözüm: beta = (X^T X)^-1 X^T y
beta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Tahminleri yap
y_pred = X_b.dot(beta)

# Cost (MSE) hesapla
mse = np.mean((y - y_pred) ** 2)
print("Least Squares MSE:", mse)

# Gerçek vs Tahmin görselleştirme
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, color="purple")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("OLS Linear Regression Tahminleri")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # 45° doğru
plt.grid(True)
import os
os.makedirs("grafikler", exist_ok=True)
plt.savefig("grafikler/ols_plot.png")   # OLS modeli için
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# [1] Veriyi Yükle
df = pd.read_csv(r"/Users/elifgungen/Documents/Laboratuvar Belgeleri/lineerRegression/data/house_price_regression_dataset.csv")



# [2] Bağımsız ve bağımlı değişkenleri ayır
X = df.drop("House_Price", axis=1).values
y = df["House_Price"].values.reshape(-1, 1)

# [3] Bias terimi (intercept) için 1'lerden oluşan sütun ekle
X_b = np.hstack([np.ones((X.shape[0], 1)), X])

# [4] Gradient Descent parametreleri
learning_rate = 0.00000001
epochs = 1000

# [5] Ağırlıkları sıfırla
m, n = X_b.shape
theta = np.zeros((n, 1))
cost_list = []

# [6] Gradient Descent Döngüsü
for epoch in range(epochs):
    y_pred = X_b.dot(theta)
    error = y_pred - y
    cost = (1 / (2 * m)) * np.sum(error ** 2)
    cost_list.append(cost)

    gradients = (1 / m) * X_b.T.dot(error)
    theta -= learning_rate * gradients

# [7] Sonuçları yazdır
print("Final Cost (Gradient Descent):", cost)
mse_gd = np.mean((y - y_pred) ** 2)
print("Gradient Descent MSE:", mse_gd)

# [8] Görselleştirme
plt.figure(figsize=(8, 5))
plt.scatter(y, y_pred, color="green")
plt.xlabel("Gerçek Fiyat")
plt.ylabel("Tahmin Edilen Fiyat")
plt.title("Gradient Descent Linear Regression Tahminleri")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.grid(True)
import os
os.makedirs("grafikler", exist_ok=True)
plt.savefig("grafikler/gd_plot.png")    # Gradient Descent modeli için
plt.show()

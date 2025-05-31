import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('TkAgg')  # PyCharm kullanıyorsan backend hatalarını önler
import matplotlib.pyplot as plt

# 1. Veriyi oku ve ön işle
df = pd.read_csv("insurance.csv")
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df_encoded.drop("charges", axis=1).values
y = df_encoded["charges"].values.reshape(-1, 1)

# 2. Normalizasyon
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y)

# 3. Eğitim/test bölmesi
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 4. Sinir Ağı Sınıfı
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        return self.Z2

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        dZ2 = (y_pred - y_true) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.5f}")
            self.backward(X, y, y_pred)

        # Grafik
        plt.plot(range(epochs), losses)
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Eğitim Sürecinde Kayıp (Loss)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("loss_plot.png")
        plt.show()



# 5. Modeli eğit
model = NeuralNetwork(input_size=8, hidden_size=50, output_size=1, learning_rate=0.01)
model.train(X_train, y_train, epochs=1000)

# 6. Test tahmini
y_pred_scaled = model.forward(X_test)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test)

# 7. Metrikler
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nTest Sonuçları:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Skoru: {r2:.4f}")

# 8. İlk 10 örneği yazdır
result_df = pd.DataFrame({
    "Gerçek Değer (charges)": y_true.flatten()[:10],
    "Tahmin Edilen Değer": y_pred.flatten()[:10]
})
print("\nİlk 10 Tahmin:")
print(result_df.to_string(index=False))

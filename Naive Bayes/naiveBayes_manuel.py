import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report


train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")


X_train = train_df.drop(columns=['deposit']).to_numpy()
y_train = train_df['deposit'].to_numpy()
X_test = test_df.drop(columns=['deposit']).to_numpy()
y_test = test_df['deposit'].to_numpy()


class CustomGaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes = np.unique(y)
        all_vars = []
        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = X_c.mean(axis=0)
            var = X_c.var(axis=0)
            all_vars.append(var)
            self.vars[c] = var 
        max_var = np.max(np.concatenate(all_vars))
        epsilon = self.var_smoothing * max_var
        
        for c in self.classes:
            self.vars[c] = self.vars[c] + epsilon

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            posteriors = []
            for c in self.classes:
                
                log_prior = np.log(self.priors[c])
                # Gaussian log olasılık yoğunluk fonksiyonu:
                # log P(x|c) = -0.5 * log(2*pi*var) - ((x - mean)^2)/(2*var)
                log_likelihood = np.sum(-0.5 * np.log(2 * np.pi * self.vars[c])
                                        - ((X[i] - self.means[c]) ** 2) / (2 * self.vars[c]))
                posterior = log_prior + log_likelihood
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)


custom_gnb = CustomGaussianNB(var_smoothing=1e-9)

start_time = time.time()
custom_gnb.fit(X_train, y_train)
train_time_custom = time.time() - start_time

start_time = time.time()
y_pred_custom = custom_gnb.predict(X_test)
test_time_custom = time.time() - start_time


accuracy_custom = accuracy_score(y_test, y_pred_custom)
conf_matrix_custom = confusion_matrix(y_test, y_pred_custom)


print("\n✅ Manuel GaussianNB Modeli Sonuçları:")
print(f"Doğruluk (Accuracy): {accuracy_custom:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom))
print("\nKarmaşıklık Matrisi:")
print(conf_matrix_custom)


plt.close('all')
fig, ax = plt.subplots(figsize=(6, 5))
cax = ax.matshow(conf_matrix_custom, cmap='Blues', alpha=0.7)
fig.colorbar(cax)
for i in range(conf_matrix_custom.shape[0]):
    for j in range(conf_matrix_custom.shape[1]):
        ax.text(x=j, y=i, s=conf_matrix_custom[i, j], va='center', ha='center', size=12, color='black')
ax.set_xlabel('Tahmin Edilen Etiket')
ax.set_ylabel('Gerçek Etiket')
ax.set_title('Custom GaussianNB Karmaşıklık Matrisi')
plt.show()

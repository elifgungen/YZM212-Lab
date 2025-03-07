import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "bank_processed.csv"  # Önceden işlenmiş veri seti
df = pd.read_csv(file_path)

features = df.drop(columns=['deposit']).columns
target = 'deposit'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_set = pd.concat([X_train, y_train], axis=1)
test_set = pd.concat([X_test, y_test], axis=1)

train_set.to_csv("train_data.csv", index=False)
test_set.to_csv("test_data.csv", index=False)

print("✅ Eğitim ve test verileri ayrı dosyalara kaydedildi.")

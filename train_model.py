import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================
# LOAD DATA
# ============================
df = pd.read_csv("data/data_stunting.csv")

# fitur & target
X = df[['umur', 'jenis_kelamin', 'tinggi_badan']]
y = df['status_gizi']

# encoding
le_jk = LabelEncoder()
X['jenis_kelamin'] = le_jk.fit_transform(X['jenis_kelamin'])

le_target = LabelEncoder()
y = le_target.fit_transform(y)

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# TRAIN MODEL
# ============================
model = GaussianNB()
model.fit(X_train, y_train)

# ============================
# EVALUASI
# ============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("=== HASIL EVALUASI ===")
print("Akurasi:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)

# ============================
# SIMPAN MODEL
# ============================
joblib.dump(model, "model/model_nb.pkl")
joblib.dump(le_jk, "model/le_jk.pkl")
joblib.dump(le_target, "model/le_target.pkl")

print("\n✅ Model berhasil dibuat & disimpan!")
import json

report = {
    "accuracy": float(acc),
    "confusion_matrix": cm.tolist(),
    "classification_report": cr
}

with open("model/report.json", "w") as f:
    json.dump(report, f)

print("✅ Report evaluasi disimpan!")
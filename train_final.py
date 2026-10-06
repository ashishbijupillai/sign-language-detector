import csv
import os

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

with open("data/samples.csv", newline="") as f:
    rows = list(csv.DictReader(f))

cols = [f"{a}{i}" for i in range(21) for a in "xy"]
X = np.array([[float(r[c]) for c in cols] for r in rows])
y = np.array([r["label"] for r in rows])

clf = KNeighborsClassifier(n_neighbors=5).fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/knn.joblib")
print(f"Trained on {len(rows)} rows. Saved models/knn.joblib")
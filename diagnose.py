import csv
from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

with open("data/samples.csv", newline="") as f:
    rows = list(csv.DictReader(f))

cols = [f"{a}{i}" for i in range(21) for a in "xy"]
X = np.array([[float(r[c]) for c in cols] for r in rows])
y = np.array([r["label"] for r in rows])
session = np.array([r["session"] for r in rows])
view = np.array([r["view"] for r in rows])
hand = np.array([r["handedness"] for r in rows])

train = np.isin(session, ["s1", "s2", "s3"])
test = session == "f1"
pred = KNeighborsClassifier(n_neighbors=5).fit(X[train], y[train]).predict(X[test])
yt, vt, ht = y[test], view[test], hand[test]

print("Accuracy on f1 by hand x view:")
for h in sorted(set(ht)):
    for v in sorted(set(vt)):
        m = (ht == h) & (vt == v)
        print(f"  {h:5s} {v:5s} {np.mean(yt[m] == pred[m]):.3f}  (n={m.sum()})")

print("\nMost common errors (true -> predicted, hand, view):")
errs = Counter(
    (a, b, h, v)
    for a, b, h, v, ok in zip(yt, pred, ht, vt, yt == pred)
    if not ok
)
for (a, b, h, v), n in errs.most_common(12):
    print(f"  {a:12s} -> {b:12s} {h:5s} {v:5s} {n}")
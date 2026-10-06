import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

LABELS = ["one", "two", "three", "four", "five", "thumbs_up", "thumbs_down", "none"]
TRAIN_SESSIONS = ["s1", "s2", "s3"]
TEST_SESSION = "f1"  # new person, never used for training

with open("data/samples.csv", newline="") as f:
    rows = list(csv.DictReader(f))

cols = [f"{a}{i}" for i in range(21) for a in "xy"]
X = np.array([[float(r[c]) for c in cols] for r in rows])
y = np.array([r["label"] for r in rows])
session = np.array([r["session"] for r in rows])
view = np.array([r["view"] for r in rows])
hand = np.array([r["handedness"] for r in rows])

MODELS = {
    "kNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "Random Forest": lambda: RandomForestClassifier(
        n_estimators=200, random_state=0, n_jobs=-1
    ),
}


def show_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    print("Confusion matrix (rows = true, columns = predicted)")
    print(" " * 13 + "".join(f"{l[:7]:>8s}" for l in LABELS))
    for lab, row in zip(LABELS, cm):
        print(f"{lab:13s}" + "".join(f"{v:8d}" for v in row))


def breakdown(y_true, y_pred, v, h):
    for name, arr in (("view", v), ("hand", h)):
        parts = []
        for val in sorted(set(arr)):
            m = arr == val
            parts.append(f"{val}: {accuracy_score(y_true[m], y_pred[m]):.3f} (n={m.sum()})")
        print(f"  by {name}: " + "   ".join(parts))


for model_name, make in MODELS.items():
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    # Leave-one-session-out
    all_true, all_pred = [], []
    for held in TRAIN_SESSIONS:
        train = np.isin(session, [s for s in TRAIN_SESSIONS if s != held])
        test = session == held
        clf = make().fit(X[train], y[train])
        pred = clf.predict(X[test])
        acc = accuracy_score(y[test], pred)
        print(f"Hold out {held}: accuracy {acc:.3f}  (n={test.sum()})")
        all_true.append(y[test])
        all_pred.append(pred)
    all_true = np.concatenate(all_true)
    all_pred = np.concatenate(all_pred)
    print(f"Leave-one-session-out overall: {accuracy_score(all_true, all_pred):.3f}")
    print()
    show_matrix(all_true, all_pred)

    # New-person test on f1
    train = np.isin(session, TRAIN_SESSIONS)
    test = session == TEST_SESSION
    clf = make().fit(X[train], y[train])
    pred = clf.predict(X[test])
    print()
    print(f"NEW PERSON TEST ({TEST_SESSION}): accuracy {accuracy_score(y[test], pred):.3f}  (n={test.sum()})")
    breakdown(y[test], pred, view[test], hand[test])
    print()
    show_matrix(y[test], pred)
    print()
    print(classification_report(y[test], pred, labels=LABELS, digits=3, zero_division=0))
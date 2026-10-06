import csv

import numpy as np

with open("data/samples.csv", newline="") as f:
    rows = list(csv.DictReader(f))

TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}


def tips(label, sessions, hand_name):
    sel = [r for r in rows if r["label"] == label and r["session"] in sessions
           and r["handedness"] == hand_name and r["view"] == "back"]
    out = {}
    for name, i in TIPS.items():
        xy = np.array([[float(r[f"x{i}"]), float(r[f"y{i}"])] for r in sel])
        out[name] = xy.mean(axis=0)
    return out, len(sel)


for label in ["three", "four", "five"]:
    train, n1 = tips(label, ["s1", "s3"], "Right")
    friend, n2 = tips(label, ["f1"], "Right")
    print(f"\n{label}: Right/back  you (s1+s3, n={n1})  vs  friend (f1, n={n2})")
    for name in TIPS:
        a, b = train[name], friend[name]
        print(f"  {name:7s} you ({a[0]:+.2f},{a[1]:+.2f})  friend ({b[0]:+.2f},{b[1]:+.2f})"
              f"  gap {np.linalg.norm(a - b):.2f}")
import csv
from collections import Counter

with open("data/samples.csv", newline="") as f:
    rows = list(csv.DictReader(f))

print("Total rows:", len(rows))

print("\nBy label (all sessions):")
for k, v in sorted(Counter(r["label"] for r in rows).items()):
    print(f"  {k:12s} {v}")

print("\nBy session / handedness / view:")
for k, v in sorted(Counter((r["session"], r["handedness"], r["view"]) for r in rows).items()):
    print(" ", k, v)

print("\nLabel x session x view:")
for lab in sorted({r["label"] for r in rows}):
    parts = []
    for s in sorted({r["session"] for r in rows}):
        for view in ("palm", "back"):
            n = sum(1 for r in rows if r["label"] == lab and r["session"] == s and r["view"] == view)
            parts.append(f"{s}/{view}:{n}")
    print(f"  {lab:12s} " + "  ".join(parts))
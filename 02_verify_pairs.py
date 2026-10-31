# BEN 2026-11-01
# Change aim: Evaluate same-person / different-person verification on pairs, with a person-disjoint split.
# File: 02_verify_pairs.py
"""Step 2: face verification. Threshold is chosen on dev people and tested on different people."""
import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from common import load_embeddings

os.makedirs("results", exist_ok=True)
names, paths, embs, _ = load_embeddings("results/embeddings.npz")
rng = random.Random(0)


# BEN 2026-11-01
# Change aim: Build balanced same-person and different-person pairs for a set of people.
# File: 02_verify_pairs.py
# Function: make_pairs
def make_pairs(people, n_pairs):
    idx = {p: [i for i, n in enumerate(names) if n == p] for p in people}
    multi = [p for p in people if len(idx[p]) >= 2]
    pos, neg = [], []
    while len(pos) < n_pairs:
        p = rng.choice(multi)
        pos.append(tuple(rng.sample(idx[p], 2)))
    while len(neg) < n_pairs:
        a, b = rng.sample(people, 2)
        neg.append((rng.choice(idx[a]), rng.choice(idx[b])))
    pairs = pos + neg
    labels = np.array([1] * len(pos) + [0] * len(neg))
    d = np.array([np.linalg.norm(embs[i] - embs[j]) for i, j in pairs])
    return d, labels


all_people = sorted(set(names.tolist()))
rng.shuffle(all_people)
cut = len(all_people) // 2
dev_people, test_people = all_people[:cut], all_people[cut:]
d_dev, y_dev = make_pairs(dev_people, 1000)
d_te, y_te = make_pairs(test_people, 1000)

# threshold: best accuracy on the dev pairs
cands = np.linspace(d_dev.min(), d_dev.max(), 200)
acc_dev = [np.mean((d_dev < t) == y_dev) for t in cands]
thr = float(cands[int(np.argmax(acc_dev))])

pred = d_te < thr
acc = float(np.mean(pred == y_te))
auc = float(roc_auc_score(y_te, -d_te))
fpr, tpr, _ = roc_curve(y_te, -d_te)
tar_far1 = float(np.interp(0.01, fpr, tpr))
std_acc = float(np.mean((d_te < 0.6) == y_te))     # dlib's usual fixed threshold 0.6
print(f"dev people {len(dev_people)}, test people {len(test_people)}")
print(f"threshold (from dev) = {thr:.3f}")
print(f"test accuracy = {acc:.4f}   ROC AUC = {auc:.4f}   TAR@FAR=1% = {tar_far1:.4f}")
print(f"test accuracy with fixed threshold 0.6 = {std_acc:.4f}")

plt.figure(figsize=(6, 3.5))
plt.hist(d_te[y_te == 1], bins=40, alpha=.7, label="same person")
plt.hist(d_te[y_te == 0], bins=40, alpha=.7, label="different people")
plt.axvline(thr, color="k", ls="--", label=f"threshold {thr:.2f}")
plt.xlabel("embedding distance"); plt.legend(); plt.tight_layout()
plt.savefig("results/distance_hist.png", dpi=150); plt.close()

plt.figure(figsize=(4.5, 4.5))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], "k:")
plt.xlabel("false accept rate"); plt.ylabel("true accept rate"); plt.title(f"ROC (AUC {auc:.3f})")
plt.tight_layout(); plt.savefig("results/roc.png", dpi=150); plt.close()

with open("results/metrics_verification.txt", "w") as f:
    f.write(f"threshold={thr:.3f}\ntest_accuracy={acc:.4f}\nroc_auc={auc:.4f}\n"
            f"tar_at_far_1pct={tar_far1:.4f}\naccuracy_fixed_0.6={std_acc:.4f}\n"
            f"dev_people={len(dev_people)}\ntest_people={len(test_people)}\n")

# BEN 2026-11-01
# Change aim: Test "is this the target person?" using a template built from enrolment photos.
# File: 03_one_person.py
"""Step 3: one-person check on public LFW photos (default: George_W_Bush).
Half of the person's photos build a template (mean embedding); the other half and photos
of other people are the test set. Public benchmark data only."""
import argparse
import os
import numpy as np
from sklearn.metrics import roc_auc_score
from common import load_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--person", default="George_W_Bush")
args = ap.parse_args()

os.makedirs("results", exist_ok=True)
names, paths, embs, _ = load_embeddings("results/embeddings.npz")
rng = np.random.RandomState(0)

target = np.where(names == args.person)[0]
others = np.where(names != args.person)[0]
if len(target) < 4:
    raise SystemExit(f"Not enough images of {args.person} in the embeddings file ({len(target)})")
rng.shuffle(target)
half = len(target) // 2
enrol, held = target[:half], target[half:]
template = embs[enrol].mean(axis=0)

d_pos = np.linalg.norm(embs[held] - template, axis=1)
d_neg = np.linalg.norm(embs[others] - template, axis=1)
y = np.r_[np.ones(len(d_pos)), np.zeros(len(d_neg))]
d = np.r_[d_pos, d_neg]
auc = roc_auc_score(y, -d)
thr = 0.6
tpr = float(np.mean(d_pos < thr))
fpr = float(np.mean(d_neg < thr))
print(f"person: {args.person}  enrol images: {len(enrol)}  held-out: {len(held)}  other images: {len(others)}")
print(f"ROC AUC = {auc:.4f}")
print(f"at distance threshold {thr}: true accept rate = {tpr:.4f}, false accept rate = {fpr:.4f}")
with open("results/metrics_one_person.txt", "w") as f:
    f.write(f"person={args.person}\nenrol={len(enrol)}\nheld_out={len(held)}\nother={len(others)}\n"
            f"roc_auc={auc:.4f}\nthreshold={thr}\ntar={tpr:.4f}\nfar={fpr:.4f}\n")

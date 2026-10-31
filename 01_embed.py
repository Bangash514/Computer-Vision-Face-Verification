# BEN 2026-11-01
# Change aim: Compute and save face embeddings for LFW images (run once, then reuse).
# File: 01_embed.py
"""Step 1: embed LFW faces.   python src/01_embed.py --data <lfw folder>
Default limits keep the run short on a free CPU; raise them for the full dataset."""
import argparse
import os
from common import find_lfw_root, list_people, build_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="folder with LFW (Kaggle download or similar)")
ap.add_argument("--max-people", type=int, default=400)
ap.add_argument("--max-per-person", type=int, default=12)
ap.add_argument("--person", default="George_W_Bush", help="extra person to keep in full for step 3")
args = ap.parse_args()

os.makedirs("results", exist_ok=True)
root = find_lfw_root(args.data)
print("LFW root:", root)
people = list_people(root, min_images=2, max_people=args.max_people,
                     max_per_person=args.max_per_person, always=(args.person,))
print(f"{len(people)} people, {sum(len(v) for v in people.values())} images")
build_embeddings(people, "results/embeddings.npz")

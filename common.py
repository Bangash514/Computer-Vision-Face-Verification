# BEN 2026-11-01
# Change aim: Shared helpers to find the LFW folder, embed faces and cache the embeddings.
# File: common.py
"""Shared helpers for face verification on the public LFW dataset (Kaggle folder layout).

LFW layout: <root>/<Person_Name>/<Person_Name>_0001.jpg ...
Embeddings come from dlib's 128-d ResNet face model (package: face_recognition).
"""
import os
import random
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png")


# BEN 2026-11-01
# Change aim: Find the folder that holds one sub-folder per person, even when Kaggle nests it.
# File: common.py
# Function: find_lfw_root
def find_lfw_root(path):
    best, best_n = None, 0
    for dirpath, dirnames, _ in os.walk(path):
        n = sum(1 for d in dirnames
                if any(f.lower().endswith(IMG_EXT) for f in os.listdir(os.path.join(dirpath, d))))
        if n > best_n:
            best, best_n = dirpath, n
    if best is None:
        raise FileNotFoundError(f"No person folders with images found under: {path}")
    return best


# BEN 2026-11-01
# Change aim: List image paths per person, with optional limits for a fast run.
# File: common.py
# Function: list_people
def list_people(root, min_images=2, max_people=None, max_per_person=None, seed=0, always=()):
    people = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(IMG_EXT))
        if len(files) >= min_images or name in always:
            people[name] = files
    rng = random.Random(seed)
    names = sorted(people)
    if max_people and len(names) > max_people:
        keep = set(always) & set(names)
        rest = [n for n in names if n not in keep]
        rng.shuffle(rest)
        names = sorted(keep | set(rest[:max_people - len(keep)]))
    out = {}
    for n in names:
        f = people[n]
        if max_per_person and len(f) > max_per_person and n not in always:
            f = rng.sample(f, max_per_person)
        out[n] = sorted(f)
    return out


# BEN 2026-11-01
# Change aim: Compute one 128-d face embedding for an image, using the face nearest the centre.
# File: common.py
# Function: embed_image
def embed_image(path):
    """Return (embedding or None, detected_flag). LFW faces are centred, so if the detector
    finds nothing, a centre box is used instead (detected_flag False)."""
    import face_recognition
    img = face_recognition.load_image_file(path)
    h, w = img.shape[:2]
    locs = face_recognition.face_locations(img)           # (top, right, bottom, left)
    detected = len(locs) > 0
    if detected:
        cy, cx = h / 2, w / 2
        loc = min(locs, key=lambda b: ((b[0] + b[2]) / 2 - cy) ** 2 + ((b[1] + b[3]) / 2 - cx) ** 2)
    else:
        loc = (int(.25 * h), int(.75 * w), int(.75 * h), int(.25 * w))
    enc = face_recognition.face_encodings(img, known_face_locations=[loc])
    return (enc[0] if enc else None), detected


# BEN 2026-11-01
# Change aim: Embed every listed image and save the result, so later steps need no re-embedding.
# File: common.py
# Function: build_embeddings
def build_embeddings(people, out_file):
    names, paths, embs, flags = [], [], [], []
    total = sum(len(v) for v in people.values())
    done = 0
    for name, files in people.items():
        for p in files:
            e, det = embed_image(p)
            done += 1
            if e is None:
                continue
            names.append(name); paths.append(p); embs.append(e); flags.append(det)
            if done % 200 == 0:
                print(f"  embedded {done}/{total}", flush=True)
    np.savez(out_file, names=np.array(names), paths=np.array(paths),
             embs=np.array(embs, dtype=np.float32), detected=np.array(flags))
    print(f"saved {len(names)} embeddings ({total - len(names)} failed), "
          f"{int(np.sum(~np.array(flags)))} used the centre-box fallback -> {out_file}")


# BEN 2026-11-01
# Change aim: Load the saved embeddings.
# File: common.py
# Function: load_embeddings
def load_embeddings(path):
    d = np.load(path, allow_pickle=False)
    return d["names"], d["paths"], d["embs"], d["detected"]

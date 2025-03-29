import os
import random
import pandas as pd

from collections import defaultdict
from tqdm import tqdm

IMAGE_DIR = "./images"
OUTPUT_CSV = "triplets.csv"
MIN_SAMPLES_PER_CLASS = 2
MAX_TRIPLETS = 10000

label_to_files = defaultdict(list)

for fname in os.listdir(IMAGE_DIR):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    if "_" not in fname:
        continue

    label = fname.split("_")[0].lower()
    label_to_files[label].append(fname)

triplets = []
labels = list(label_to_files.keys())

for label in tqdm(labels, desc="Building triplets"):
    files = label_to_files[label]
    if len(files) < MIN_SAMPLES_PER_CLASS:
        continue

    negative_labels = [l for l in labels if l != label]
    for anchor in files:
        positive = random.choice([f for f in files if f != anchor])
        negative_label = random.choice(negative_labels)
        negative = random.choice(label_to_files[negative_label])
        triplets.append((anchor, positive, negative))

        if len(triplets) >= MAX_TRIPLETS:
            break
    if len(triplets) >= MAX_TRIPLETS:
        break

df = pd.DataFrame(triplets, columns=["anchor", "positive", "negative"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved {len(triplets)} triplets to {OUTPUT_CSV}")
import pandas as pd

TRIPLET_CSV = "./triplet/triplets.csv"
OUTPUT_TXT = "./triplet/triplet_images.txt"

# Load triplet combinations
df = pd.read_csv(TRIPLET_CSV)

# Collect all unique filenames from anchor, positive, and negative columns
all_images = pd.unique(df[["anchor", "positive", "negative"]].values.ravel())

# Sort and write to text file
with open(OUTPUT_TXT, "w") as f:
    for img in sorted(all_images):
        f.write(img + "\n")

print(f"Saved {len(all_images)} image filenames to {OUTPUT_TXT}")


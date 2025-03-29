import numpy as np
import faiss
import os

# Paths
EMBEDDING_NPY = "./npy/image_embeddings_pca256.npy"
IMAGE_LIST_TXT = "./triplet/triplet_images.txt"
OUTPUT_INDEX = "./faiss/image_index_pca.faiss"
OUTPUT_FILENAMES = "./faiss/image_filenames.npy"
os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)

# Load
reduced = np.load(EMBEDDING_NPY).astype("float32")
with open(IMAGE_LIST_TXT) as f:
    image_files = [line.strip() for line in f.readlines()]
assert len(image_files) == reduced.shape[0]

# Build index
index = faiss.IndexFlatL2(reduced.shape[1])
index.add(reduced)
faiss.write_index(index, OUTPUT_INDEX)
np.save(OUTPUT_FILENAMES, np.array(image_files))

print(f"FAISS PCA index saved to: {OUTPUT_INDEX}")


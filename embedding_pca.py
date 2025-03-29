import numpy as np
from sklearn.decomposition import PCA
import joblib
import os

INPUT_EMBEDDING = "./npy/image_embeddings.npy"
OUTPUT_REDUCED = "./npy/image_embeddings_pca256.npy"
OUTPUT_PCA_MODEL = "./npy/pca_model.joblib"
N_COMPONENTS = 256

# Load original embeddings
embeddings = np.load(INPUT_EMBEDDING)

# Fit PCA
pca = PCA(n_components=N_COMPONENTS)
reduced = pca.fit_transform(embeddings)

# Save results
np.save(OUTPUT_REDUCED, reduced)
joblib.dump(pca, OUTPUT_PCA_MODEL)

print(f"Saved reduced embeddings: {reduced.shape} → {OUTPUT_REDUCED}")
print(f"Saved PCA model to: {OUTPUT_PCA_MODEL}")

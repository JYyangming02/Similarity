import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.Dino import ImageEmbeddingModel

IMAGE_DIR = "./images"
IMAGE_LIST_TXT = "./triplet/triplet_images.txt"
MODEL_PATH = "./models/triplet_dino.pth"
OUTPUT_NPY = "./npy/image_embeddings.npy"
BATCH_SIZE = 32
EMBEDDING_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(IMAGE_LIST_TXT) as f:
    image_files = [line.strip() for line in f.readlines()]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

model = ImageEmbeddingModel(output_dim=EMBEDDING_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_embeddings = []

with torch.no_grad():
    for i in range(0, len(image_files), BATCH_SIZE):
        batch = image_files[i:i+BATCH_SIZE]
        imgs = []

        for fname in batch:
            img_path = os.path.join(IMAGE_DIR, fname)
            img = Image.open(img_path).convert("RGB")
            imgs.append(transform(img))

        img_tensor = torch.stack(imgs).to(DEVICE)
        embeddings = model(img_tensor)  # [B, 768]
        all_embeddings.append(embeddings.cpu().numpy())

final_embeddings = np.concatenate(all_embeddings, axis=0)
np.save(OUTPUT_NPY, final_embeddings)
print(f"Saved embeddings: {final_embeddings.shape} → {OUTPUT_NPY}")
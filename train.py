import os
import torch
import torch.nn as nn
 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.cosine_triplet_loss import CosineTripletLoss
from models.Dino import ImageEmbeddingModel
from triplet.triplet_dataset import TripletImageDataset

TRIPLET_CSV = "./triplet/triplets.csv"
IMAGE_DIR = "./images"
BATCH_SIZE = 32
EPOCHS = 10
EMBEDDING_DIM = 768
LR = 1e-4
MODEL_SAVE_PATH = "./models/triplet_dino.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    dataset = TripletImageDataset(TRIPLET_CSV, IMAGE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = ImageEmbeddingModel(output_dim=EMBEDDING_DIM).to(DEVICE)
    criterion = CosineTripletLoss(margin=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    best_avg_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"[Epoch {epoch}]")
        for anchor, positive, negative in pbar:
            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} finished. Avg loss: {avg_loss:.4f}")
        
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("Saved best model.")
        else:
            epochs_no_improve += 1

if __name__ == "__main__":
    main()

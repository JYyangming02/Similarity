import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TripletImageDataset(Dataset):
    def __init__(self, triplet_csv_path, image_dir, transform=None):
        """
        Args:
            triplet_csv_path (str): Path to the triplets.csv file
            image_dir (str): Directory containing image files
            transform (callable, optional): Optional transform to be applied to images
        """
        import pandas as pd
        self.image_dir = image_dir
        self.triplets = pd.read_csv(triplet_csv_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        row = self.triplets.iloc[idx]
        anchor_path = os.path.join(self.image_dir, row["anchor"])
        positive_path = os.path.join(self.image_dir, row["positive"])
        negative_path = os.path.join(self.image_dir, row["negative"])

        anchor_img = Image.open(anchor_path).convert("RGB")
        positive_img = Image.open(positive_path).convert("RGB")
        negative_img = Image.open(negative_path).convert("RGB")

        return (
            self.transform(anchor_img),
            self.transform(positive_img),
            self.transform(negative_img),
        )

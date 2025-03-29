import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights

class ImageEmbeddingModel(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        # Load pretrained DINO ViT-B/16
        weights = ViT_B_16_Weights.DEFAULT
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Identity()  # Remove classification head

        # Output dim from ViT-B/16 is 768 by default
        self.embedding = nn.Sequential(
            nn.Linear(768, 768),  # Can adjust input/output here if needed
            nn.ReLU(),
            nn.Linear(768, output_dim)
        )

    def forward(self, x):
        features = self.backbone(x)           # [B, 768]
        embedding = self.embedding(features)  # [B, output_dim]
        return embedding

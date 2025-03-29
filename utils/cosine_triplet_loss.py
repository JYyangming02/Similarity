import torch
import torch.nn as nn
import torch.nn.functional as F

class CosineTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negative = F.normalize(negative, dim=1)

        pos_sim = F.cosine_similarity(anchor, positive)
        neg_sim = F.cosine_similarity(anchor, negative)

        # Loss: max(0, margin + sim_neg - sim_pos)
        loss = F.relu(self.margin + neg_sim - pos_sim).mean()
        return loss
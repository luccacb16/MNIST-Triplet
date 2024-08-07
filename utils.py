import numpy as np
import argparse

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomAffine
from torch.utils.data import Dataset
import torch.nn.functional as F

# Transform com Data Augmentation
aug_transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,)),
    RandomRotation(10),
    RandomAffine(degrees=5, translate=(0.05, 0.05))
])

# Transform comum
simple_transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# --------------------------------------------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, images_df, transform=simple_transform, tensor_type=torch.float16):
        self.labels = images_df['label'].values
        self.pixel_values = images_df.drop(columns=['label']).values
        self.transform = transform
        self.tensor_type = tensor_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.pixel_values[idx].reshape(28, 28).astype(np.float16)
        label = self.labels[idx]
        
        image = torch.tensor(image).unsqueeze(0)
        
        image = image.expand(3, -1, -1)
        
        return image.to(self.tensor_type), label

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, triplets, embeddings):
        triplets = triplets.long()
        
        anchor_embeddings = embeddings[triplets[:, 0]]
        positive_embeddings = embeddings[triplets[:, 1]]
        negative_embeddings = embeddings[triplets[:, 2]]
        
        positive_distances = torch.norm(anchor_embeddings - positive_embeddings, p=2, dim=1)
        negative_distances = torch.norm(anchor_embeddings - negative_embeddings, p=2, dim=1)
        
        losses = F.relu(positive_distances - negative_distances + self.margin)
        return losses.mean()

# --------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural com triplet loss")
    parser.add_argument('--num_images', type=int, default=4992, help='Número de imagens (default: 4992)')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch (default: 128)')
    parser.add_argument('--epochs', type=int, default=8, help='Número de epochs (default: 8)')
    parser.add_argument('--margin', type=float, default=0.2, help='Margem para triplet loss (default: 0.2)')
    parser.add_argument('--num_workers', type=int, default=0, help='Número de workers para o DataLoader (default: 0)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Caminho para o dataset (default: ./data)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    
    return parser.parse_args()
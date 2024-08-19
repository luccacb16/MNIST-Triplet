import numpy as np
import argparse

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomAffine
from torch.utils.data import Dataset, Sampler
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
    def __init__(self, images_df, transform=simple_transform, dtype=torch.float16):
        self.labels = images_df['label'].values
        self.pixel_values = images_df.drop(columns=['label']).values
        self.transform = transform
        self.dtype = dtype

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_data = self.pixel_values[idx]

        if image_data.size != 784:  # Verifique se o tamanho é 784
            raise ValueError(f"Expected 784 elements, got {image_data.size}")

        image = image_data.reshape(28, 28).astype(np.float16)
        label = self.labels[idx]

        image = torch.tensor(image).unsqueeze(0)  # Adiciona um canal
        return image.to(self.dtype), label
    
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        print(f'Labels: {len(labels)}')
        print(f'n_classes: {n_classes}')
        print(f'n_samples: {n_samples}')
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples

        self.used_labels_indices = {label: 0 for label in self.labels_set}

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size <= len(self.labels):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                start = self.used_labels_indices[class_]
                end = start + self.n_samples
                if end > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    start = 0
                    end = self.n_samples
                indices.extend(self.label_to_indices[class_][start:end])
                self.used_labels_indices[class_] = end
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.labels) // self.batch_size
class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, triplets, embeddings):
        triplets = triplets.long()
        
        anchor_embeddings = embeddings[triplets[:, 0]]
        positive_embeddings = embeddings[triplets[:, 1]]
        negative_embeddings = embeddings[triplets[:, 2]]
        
        # Calculando as distâncias como a soma dos quadrados das diferenças
        positive_distances = (anchor_embeddings - positive_embeddings).pow(2).sum(dim=1)
        negative_distances = (anchor_embeddings - negative_embeddings).pow(2).sum(dim=1)
        
        # Calculando as perdas com a margem, considerando as distâncias quadradas
        losses = F.relu(positive_distances - negative_distances + self.margin)
        return losses.mean()

# --------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural com triplet loss")
    parser.add_argument('--num_images', type=int, default=4992, help='Número de imagens (default: 4992)')
    parser.add_argument('--batch_size', type=int, default=128, help='Tamanho do batch (default: 128)')
    parser.add_argument('--epochs', type=int, default=8, help='Número de epochs (default: 8)')
    parser.add_argument('--margin', type=float, default=0.5, help='Margem para triplet loss (default: 0.5)')
    parser.add_argument('--num_workers', type=int, default=0, help='Número de workers para o DataLoader (default: 0)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Caminho para o dataset (default: ./data)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    
    return parser.parse_args()
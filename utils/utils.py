import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomAffine
from torch.utils.data import Dataset, Sampler
import torch.nn.functional as F

# Data Augmentation Transform
aug_transform = Compose([
    ToTensor(),
    RandomAffine(degrees=5, translate=(0.05, 0.05)),
    RandomRotation(10),
    Normalize((0.5,), (0.5,))
])

# Simple Transform
simple_transform = Compose([
    ToTensor(),
    Normalize((0.5,), (0.5,))
])

# --------------------------------------------------------------------------------------------------------

class TripletDataset(Dataset):
    def __init__(self, images_df, transform=None):
        self.labels = images_df['label'].values
        self.pixel_values = images_df.drop(columns=['label']).values.astype(np.uint8)
        self.transform = transform

    def __getitem__(self, idx):
        image_data = self.pixel_values[idx].reshape(28, 28)
        
        if self.transform:
            image_data = self.transform(image_data)
            
        label = self.labels[idx]
        return image_data, torch.tensor(label)

    def __len__(self):
        return len(self.labels)

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, batch_size):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        
        for label_indices in self.label_to_indices.values():
            np.random.shuffle(label_indices)
        
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_samples = batch_size // n_classes

    def __iter__(self):
        self.count = 0
        total_indices = len(self.labels)
        while self.count + self.batch_size <= total_indices:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(np.random.choice(self.label_to_indices[class_], self.n_samples, replace=False))
            
            if len(indices) < self.batch_size:
                extra_needed = self.batch_size - len(indices)
                all_indices = np.arange(total_indices)
                extra_indices = np.random.choice(all_indices, extra_needed, replace=False)
                indices.extend(extra_indices)
            
            np.random.shuffle(indices)
            yield indices
            self.count += len(indices)

    def __len__(self):
        return len(self.labels) // self.batch_size

class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor_embeddings, positive_embeddings, negative_embeddings):
        pos_diff = anchor_embeddings - positive_embeddings
        neg_diff = anchor_embeddings - negative_embeddings

        positive_distances = torch.norm(pos_diff, p=2, dim=1)
        negative_distances = torch.norm(neg_diff, p=2, dim=1)

        losses = F.relu(positive_distances - negative_distances + self.margin)

        return losses.mean()

# --------------------------------------------------------------------------------------------------------

# Validação

def get_val_triplets(test_df: pd.DataFrame, n: int = 128) -> torch.Tensor:
    """
    Gera n triplets (anchor, positive, negative) a partir do conjunto de teste.

    Args:
        test_df (pd.DataFrame): DataFrame contendo as colunas 'label' e 784 colunas de pixels (MNIST).
        n (int): Número de triplets a serem gerados. Padrão é 128.

    Returns:
        torch.Tensor: Tensor de forma (n, 3) contendo os índices dos triplets.
        torch.Tensor: Tensor de forma (n,) contendo os rótulos dos triplets.
    """
    
    labels = test_df['label'].values
    num_samples = len(test_df)

    triplets = []
    for _ in range(n):
        # Selecionar âncora, positivo e negativo aleatoriamente
        anchor_idx = np.random.randint(0, num_samples)
        anchor_label = labels[anchor_idx]

        positive_idxs = np.where(labels == anchor_label)[0]
        positive_idxs = positive_idxs[positive_idxs != anchor_idx]
        positive_idx = np.random.choice(positive_idxs)

        negative_idxs = np.where(labels != anchor_label)[0]
        negative_idx = np.random.choice(negative_idxs)

        triplets.append((anchor_idx, positive_idx, negative_idx))

    return torch.tensor(triplets, dtype=torch.int16)

class ValTripletsDataset(Dataset):
    def __init__(self, test_df: pd.DataFrame, transform=None, n_triplets: int = 128):
        self.test_data = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.test_labels = torch.tensor(test_df['label'].values, dtype=torch.int16)
        self.transform = transform
        
        self.triplets = get_val_triplets(test_df, n_triplets)

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        anchor_idx, pos_idx, neg_idx = self.triplets[index]
        
        anchor_img = self.test_data[anchor_idx]
        pos_img = self.test_data[pos_idx]
        neg_img = self.test_data[neg_idx]
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)
        
        return (anchor_img, pos_img, neg_img), torch.tensor([anchor_idx, pos_idx, neg_idx], dtype=torch.int16) 
    
def calc_val_loss(model, val_loader, triplet_loss, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for (anchors, positives, negatives), _ in val_loader:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
        
            # Gerar os embeddings
            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            negative_embeddings = model(negatives)

            # Calcular a perda de triplet
            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
            total_loss += loss.item()
            num_batches += 1

    val_loss = total_loss / num_batches
    model.train()
    
    return val_loss

def save_losses(train_losses: list, val_losses: list, save_path: str = './imgs/'):
    """
    Salva os valores das perdas de treino e validação em um gráfico.

    Args:
        train_losses (list): Lista contendo os valores das perdas de treino.
        val_losses (list): Lista contendo os valores das perdas de validação.
        save_path (str): Caminho para salvar a imagem. Padrão é './imgs/'.
    """
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(save_path + 'losses.png')

# --------------------------------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Treinar a rede neural com triplet loss")
    parser.add_argument('--num_images', type=int, default=59904, help='Número de imagens (default: 59904)')
    parser.add_argument('--batch_size', type=int, default=256, help='Tamanho do batch (default: 256)')
    parser.add_argument('--epochs', type=int, default=20, help='Número de epochs (default: 20)')
    parser.add_argument('--margin', type=float, default=1, help='Margem para triplet loss (default: 1.0)')
    parser.add_argument('--num_workers', type=int, default=0, help='Número de workers para o DataLoader (default: 0)')
    parser.add_argument('--data_path', type=str, default='./data/', help='Caminho para o dataset (default: ./data/)')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/', help='Caminho para salvar os checkpoints (default: ./checkpoints/)')
    parser.add_argument('--change_mining_strategy', type=int, default=0, help='Mudar a estratégia de mineração de triplets de semi-hard para hard')
    parser.add_argument('--lr_step', type=int, default=10, help='Número de epochs para reduzir a taxa de aprendizado')
        
    return parser.parse_args()
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

from models.omoindrot import Omoindrot
from models.mini import MiniNet

from data import download_dataset
from utils import parse_args, simple_transform, TripletDataset, BalancedBatchSampler, TripletLoss
from triplet_selection import semi_hard_triplet_mining, hard_negative_triplet_mining

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
dtype = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        dtype = torch.float16
        
EMB_SIZE = 64
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    epochs: int = 8,
    margin: float = 0.5,
    checkpoint_path: str = './checkpoints/',
    device: str = 'cuda',
):
    model.train()
    
    losses = []
    for epoch in range(epochs):
        total_triplets = 0
        total_batches = 0
        for imgs, labels in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", unit='batch'):
            total_triplets += len(imgs)
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                embeddings = model(imgs)
                
                # Trocar a estratégia de mining após 40% das epochs
                if (epoch+1) < (epochs+1) * 0.4:
                    triplets = semi_hard_triplet_mining(embeddings=embeddings, labels=labels, margin=margin, device=device, hardest=False)
                else:
                    triplets = hard_negative_triplet_mining(embeddings=embeddings, labels=labels, device=device)
                    
                loss = triplet_loss(triplets, embeddings)
                
                total_triplets += triplets.shape[1]
                total_batches += 1
                
            loss.backward()
            optimizer.step()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item()}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
        
# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    
    num_images = args.num_images
    batch_size = args.batch_size
    epochs = args.epochs
    margin = args.margin
    num_workers = args.num_workers
    data_path = args.data_path
    CHECKPOINT_PATH = args.checkpoint_path
    
    # Cria o diretório para salvar os checkpoints
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    
    # Baixa o dataset MNIST
    download_dataset(data_path=data_path)
    
    train_df = pd.read_csv(os.path.join(data_path, 'mnist_train.csv'))
    if num_images > len(train_df):
        num_images = len(train_df)
    train_df = train_df.sample(n=num_images, replace=True).reset_index(drop=True)
    
    print()
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {dtype}\n')
    
    triplet_dataset = TripletDataset(train_df, transform=simple_transform, dtype=dtype)
    
    #n_classes = len(np.unique(triplet_dataset.labels))
    #n_samples = batch_size // n_classes
    #sampler = BalancedBatchSampler(triplet_dataset.labels, n_classes=n_classes, n_samples=n_samples)
    
    dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    triplet_loss = TripletLoss(margin=margin)
    
    model = MiniNet(emb_size=EMB_SIZE).to(device)
    model = torch.compile(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    train(
        model           = model,
        dataloader      = dataloader,
        optimizer       = optimizer,
        triplet_loss    = triplet_loss,
        epochs          = epochs,
        checkpoint_path = CHECKPOINT_PATH,
        margin          = margin,
        device          = device,
    )
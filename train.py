import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

from models.omoindrot import Omoindrot

from data import download_dataset
from utils.utils import parse_args, simple_transform, TripletDataset, BalancedBatchSampler, TripletLoss, ValTripletsDataset, calc_val_loss, save_losses
from triplet_mining import semi_hard_triplet_mining, hard_negative_triplet_mining

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
dtype = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        dtype = torch.float16
        
EMB_SIZE = 64
CHANGE_MINING_STRATEGY = 0.4
N_VAL_TRIPLETS = 128
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    scheduler: torch.optim.lr_scheduler = None,
    epochs: int = 20,
    margin: float = 0.7,
    checkpoint_path: str = './checkpoints/',
    device: str = 'cuda',
):
    
    train_losses = []
    val_losses = []
    
    model.train()
    
    for epoch in range(epochs):
        for imgs, labels in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", unit='batch'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                embeddings = model(imgs)
                
                # Troca a estratégia de mineração de triplets após CHANGE_MINING_STRATEGY% das epochs
                if (epoch+1) < int((epochs+1) * CHANGE_MINING_STRATEGY):
                    triplets = semi_hard_triplet_mining(embeddings=embeddings, labels=labels, margin=margin, device=device, hardest=False)
                else:
                    triplets = hard_negative_triplet_mining(embeddings=embeddings, labels=labels, device=device)
                
                loss = triplet_loss(triplets, embeddings)
                
            loss.backward()
            optimizer.step()
            
        if scheduler is not None:
            scheduler.step()
            
        # Validation loss
        val_loss = calc_val_loss(model, val_dataloader, triplet_loss, device)
        val_losses.append(val_loss)
        train_losses.append(loss.item())
            
        print(f"Epoch [{epoch+1}/{epochs}] | loss: {loss.item()} | val_loss: {val_loss} | LR: {optimizer.param_groups[0]['lr']:.0e}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
        
    return train_losses, val_losses
        
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
    
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)
    
    # MNIST dataset download
    download_dataset(data_path=data_path)
    
    print()
    print(f'Device: {device}')
    print(f'Device name: {torch.cuda.get_device_name()}')
    print(f'Using tensor type: {dtype}\n')
    
    # Carregando datasets
    train_df = pd.read_csv(os.path.join(data_path, 'mnist_train.csv'))
    if num_images > len(train_df):
        num_images = len(train_df)
    train_df = train_df.sample(n=num_images).reset_index(drop=True)
    
    # Loader de validação
    test_df = pd.read_csv(os.path.join(data_path, 'mnist_test.csv'))
    val_triplets = ValTripletsDataset(test_df, n_triplets=N_VAL_TRIPLETS)
    val_dataloader = DataLoader(val_triplets, batch_size=N_VAL_TRIPLETS, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    # Loader de treino
    triplet_dataset = TripletDataset(train_df, transform=simple_transform)
    
    n_classes = len(np.unique(triplet_dataset.labels))
    sampler = BalancedBatchSampler(triplet_dataset.labels, n_classes=n_classes, batch_size=batch_size)
    
    dataloader = DataLoader(triplet_dataset, batch_sampler=sampler,
                            pin_memory=True, num_workers=num_workers)
    
    # Loss
    triplet_loss = TripletLoss(margin=margin)
    
    # Modelo
    model = Omoindrot().to(device)
    model = torch.compile(model)
    
    # Otimizador e scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses, val_losses = train(
        model           = model,
        dataloader      = dataloader,
        val_dataloader  = val_dataloader,
        optimizer       = optimizer,
        triplet_loss    = triplet_loss,
        scheduler       = scheduler,
        epochs          = epochs,
        checkpoint_path = CHECKPOINT_PATH,
        margin          = margin,
        device          = device,
    )
    
    # Salva a imagem com os resultados
    save_losses(train_losses, val_losses, os.path.join(CHECKPOINT_PATH, 'losses.png'))
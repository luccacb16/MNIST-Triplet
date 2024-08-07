import pandas as pd
from tqdm import tqdm
import os

import torch
from torch.utils.data import DataLoader

from models.mini_inception_resnet_v1 import MiniInceptionResNetV1 
from data import download_dataset
from utils import parse_args, aug_transform, simple_transform, TripletDataset, TripletLoss
from triplet_selection import hard_negative_triplet_selection

torch.set_float32_matmul_precision('high')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor_type = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        tensor_type = torch.float16
        
EMB_SIZE = 128
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    epochs: int = 8,
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
            
            with torch.cuda.amp.autocast(dtype=tensor_type):
                embeddings = model(imgs)
                triplets = hard_negative_triplet_selection(embeddings=embeddings, labels=labels, device=device)
                loss = triplet_loss(triplets, embeddings)
                total_triplets += triplets.shape[1]
                total_batches += 1
                
            loss.backward()
            optimizer.step()
            
        losses.append(loss.item())
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss.item()} | Média de triplets: {total_triplets / total_batches:.0f}")
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'epoch_{epoch+1}.pt'))
        
    return losses
            
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
    print(f'Using tensor type: {tensor_type}\n')
    
    triplet_dataset = TripletDataset(train_df, transform=simple_transform, tensor_type=tensor_type)
    triplet_loss = TripletLoss(margin=margin)
    dataloader = DataLoader(triplet_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    model = MiniInceptionResNetV1(emb_size=EMB_SIZE).to(device)
    model = torch.compile(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    losses = train(
        model           = model,
        dataloader      = dataloader,
        optimizer       = optimizer,
        triplet_loss    = triplet_loss,
        epochs          = epochs,
        checkpoint_path = CHECKPOINT_PATH,
        device          = device,
    )
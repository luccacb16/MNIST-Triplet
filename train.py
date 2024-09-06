import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import wandb
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader

from models.omoindrot import Omoindrot

from data import download_dataset
from utils.utils import parse_args, simple_transform, TripletDataset, BalancedBatchSampler, TripletLoss, ValTripletsDataset, calc_val_loss
from utils.eval_utils import get_pairs, MNISTPairsDataset, calc_accuracy 
from triplet_mining import semi_hard_triplet_mining, hard_negative_triplet_mining

load_dotenv()

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
dtype = torch.bfloat16
if torch.cuda.is_available():
    gpu_properties = torch.cuda.get_device_properties(0)

    if gpu_properties.major < 8:
        dtype = torch.float16
        
EMB_SIZE = 64
CHANGE_MINING_STRATEGY = 8
N_VAL_TRIPLETS = 128
        
# --------------------------------------------------------------------------------------------------------

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    val_dataloader: DataLoader,
    acc_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    triplet_loss: TripletLoss,
    scheduler: torch.optim.lr_scheduler = None,
    epochs: int = 20,
    margin: float = 0.7,
    checkpoint_path: str = './checkpoints/',
    device: str = 'cuda',
):
    
    model.train()
    
    for epoch in range(epochs):
        for imgs, labels in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", unit='batch'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(dtype=dtype):
                embeddings = model(imgs)
                
                # Troca a estratégia de mineração de triplets após CHANGE_MINING_STRATEGY% das epochs
                if CHANGE_MINING_STRATEGY > 0 and (epoch+1) > CHANGE_MINING_STRATEGY:
                    triplets = hard_negative_triplet_mining(embeddings=embeddings, labels=labels, device=device)
                else:
                    triplets = semi_hard_triplet_mining(embeddings=embeddings, labels=labels, margin=margin, device=device, hardest=False)
                
                anchor_embeddings = embeddings[triplets[:, 0]]
                positive_embeddings = embeddings[triplets[:, 1]]
                negative_embeddings = embeddings[triplets[:, 2]]
                
                loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)
                
            loss.backward()
            optimizer.step()
            
            val_loss = calc_val_loss(model, val_dataloader, triplet_loss, device)
            
            # Log
            wandb.log({'epoch': epoch, 'train_loss': loss.item(), 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr']})
        
        # Acurácia
        epoch_accuracy = calc_accuracy(model, acc_dataloader, device)
        wandb.log({'epoch': epoch, 'accuracy': epoch_accuracy})
        
        if scheduler is not None:
            scheduler.step()
            
        print(f"Epoch [{epoch+1}/{epochs}] | accuracy: {epoch_accuracy:4f} | loss: {loss.item():.6f} | val_loss: {val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.0e}")
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
    CHANGE_MINING_STRATEGY = args.change_mining_strategy
    LR_STEP = args.lr_step
    
    config = {
        'num_images': num_images,
        'batch_size': batch_size,
        'epochs': epochs,
        'margin': margin,
        'num_workers': num_workers,
        'data_path': data_path,
        'checkpoint_path': CHECKPOINT_PATH,
        'change_mining_strategy': CHANGE_MINING_STRATEGY,
        'lr_step': LR_STEP,
    }
    
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(project='mnist-triplet-loss', config=config)
    
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
    
    # Loaders de validação e acurácia
    test_df = pd.read_csv(os.path.join(data_path, 'mnist_test.csv'))
    val_triplets = ValTripletsDataset(test_df, n_triplets=N_VAL_TRIPLETS)
    val_dataloader = DataLoader(val_triplets, batch_size=N_VAL_TRIPLETS, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    acc_pairs = get_pairs(test_df, N_VAL_TRIPLETS)
    acc_dataset = MNISTPairsDataset(acc_pairs, transform=simple_transform)
    acc_dataloader = DataLoader(acc_dataset, batch_size=N_VAL_TRIPLETS, shuffle=False, pin_memory=True, num_workers=num_workers)
    
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
    wandb.watch(model, log='all', log_freq=100)
    
    # Otimizador e scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1 if epoch >= LR_STEP else 1.0)
    
    train(
        model           = model,
        dataloader      = dataloader,
        val_dataloader  = val_dataloader,
        acc_dataloader  = acc_dataloader,
        optimizer       = optimizer,
        triplet_loss    = triplet_loss,
        scheduler       = scheduler,
        epochs          = epochs,
        checkpoint_path = CHECKPOINT_PATH,
        margin          = margin,
        device          = device,
    )
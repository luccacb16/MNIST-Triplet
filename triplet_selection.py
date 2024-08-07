import torch
import numpy as np

def hard_positive_triplet_selection(embeddings: torch.Tensor, labels: torch.Tensor, device: str = 'cuda'):
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    triplets = []
    labels = labels.to(device)
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        anchor_distance = distance_matrix[i]
        
        # Selecionar hardest positive (mesma classe, maior distância)
        positive_mask = (labels == anchor_label)
        positive_distances = anchor_distance * positive_mask.float()
        positive_distances[i] = -1  # Ignorar o próprio âncora
        hardest_positive_index = torch.argmax(positive_distances)
        
        # Selecionar hardest negative (classe diferente, menor distância)
        negative_mask = (labels != anchor_label)
        negative_distances = anchor_distance + (1 - negative_mask.float()) * 1e6
        hardest_negative_index = torch.argmin(negative_distances)
        
        triplets.append((i, hardest_positive_index.item(), hardest_negative_index.item()))
    
    return torch.tensor(triplets, dtype=torch.long, device=device)

def hard_negative_triplet_selection(embeddings: torch.Tensor, labels: torch.Tensor, device: str = 'cuda'):
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    triplets = []
    labels = labels.to(device)
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        
        positive_indices = (labels == anchor_label).nonzero().squeeze()
        positive_indices = positive_indices[positive_indices != i]
        if len(positive_indices) > 0:
            random_positive_index = positive_indices[np.random.randint(0, len(positive_indices))].item()
        else:
            continue 
        
        positive_distance = distance_matrix[random_positive_index]
        negative_mask = (labels != anchor_label)
        negative_distances = positive_distance + (1 - negative_mask.float()) * 1e6
        hardest_negative_index = torch.argmin(negative_distances)
        
        triplets.append((i, random_positive_index, hardest_negative_index.item()))
    
    return torch.tensor(triplets, dtype=torch.long, device=device)

def semi_hard_triplet_mining(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.2, device: str = 'cuda'):
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    triplets = []
    labels = labels.to(device)
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        anchor_distance = distance_matrix[i]

        # Buscar todos os positivos para cada âncora
        positive_indices = torch.where(labels == anchor_label)[0]
        negative_indices = torch.where(labels != anchor_label)[0]

        for positive_index in positive_indices:
            if i == positive_index:
                continue  # Ignorar o próprio âncora

            positive_distance = anchor_distance[positive_index]

            # Buscar todos os negativos que atendam a condição d(a, p) < d(a, n) < d(a, p) + margin
            valid_negative_distances = anchor_distance[negative_indices]
            condition_mask = (valid_negative_distances > positive_distance) & \
                             (valid_negative_distances < positive_distance + margin)

            valid_negatives = negative_indices[condition_mask]

            if len(valid_negatives) > 0:
                # Escolher o negative que está mais próximo do limite superior da condição
                hardest_negative_index = valid_negatives[torch.argmin(valid_negative_distances[condition_mask])]
                triplets.append((i, positive_index.item(), hardest_negative_index.item()))
    
    if len(triplets) > 0:
        return torch.tensor(triplets, dtype=torch.long, device=device)
    else:
        return torch.tensor([], dtype=torch.long, device=device)

def batch_random_triplet_selection(labels, num_triplets, device='cuda'):
    triplets = []
    
    # Move labels to CPU for processing
    labels_cpu = labels.cpu()

    for _ in range(num_triplets):
        anchor_index = np.random.randint(0, len(labels))
        anchor_label = labels_cpu[anchor_index]  # Use labels_cpu here to ensure everything is on the CPU

        positive_indices = torch.where(labels_cpu == anchor_label)[0]
        negative_indices = torch.where(labels_cpu != anchor_label)[0]

        if len(positive_indices) > 0 and len(negative_indices) > 0:
            positive_index = np.random.choice(positive_indices)
            negative_index = np.random.choice(negative_indices)
            triplets.append((anchor_index, positive_index.item(), negative_index.item()))

    return torch.tensor(triplets, dtype=torch.long, device=device)
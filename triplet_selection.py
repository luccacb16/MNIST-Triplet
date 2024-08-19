import torch
import numpy as np

def hard_negative_triplet_mining(embeddings: torch.Tensor, labels: torch.Tensor, device: str = 'cuda'):
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

def semi_hard_triplet_mining(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.5, device: str = 'cuda'):
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    triplets = []
    labels = labels.to(device)
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        anchor_distances = distance_matrix[i]

        positive_indices = torch.where(labels == anchor_label)[0]
        positive_indices = positive_indices[positive_indices != i]

        if len(positive_indices) == 0:
            continue
        
        random_positive_index = np.random.choice(positive_indices.cpu().numpy())
        positive_distance_squared = anchor_distances[random_positive_index]

        negative_indices = torch.where(labels != anchor_label)[0]
        valid_negative_distances = anchor_distances[negative_indices]

        semi_hard_mask = (valid_negative_distances > positive_distance_squared) & \
                         (valid_negative_distances < positive_distance_squared + margin)
        
        valid_negatives = negative_indices[semi_hard_mask]
        
        if len(valid_negatives) > 0:
            # Escolher o negative mais desafiador dentro dos semi-hard
            hardest_semi_hard_negative_index = valid_negatives[torch.argmin(valid_negative_distances[semi_hard_mask])]
            triplets.append((i, random_positive_index, hardest_semi_hard_negative_index.item()))
        else:
            # Se nenhum semi-hard negative, escolher o hardest negative
            hardest_negative_index = negative_indices[torch.argmin(valid_negative_distances)]
            triplets.append((i, random_positive_index, hardest_negative_index.item()))

    return torch.tensor(triplets, dtype=torch.long, device=device)
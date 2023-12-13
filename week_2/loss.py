import torch
import torch.nn.functional as F
# Loss function
def contrastive_loss(query_vector, sentence_vector, label, margin=0.5):
    # Cosine similarity
    sim = F.cosine_similarity(query_vector, sentence_vector, dim=1)

    # Loss computation
    loss = (1 - label) * torch.pow(sim, 2) + label * torch.pow(F.relu(margin - sim), 2)
    return loss.mean()
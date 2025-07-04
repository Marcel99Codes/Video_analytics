import torch
import torch.nn.functional as F

def cosine_similarity(a, b):
    return F.cosine_similarity(a, b)

def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2N x feature_dim
    z = F.normalize(z, dim=1)
    sim = torch.matmul(z, z.T) / temperature  # 2N x 2N similarity matrix
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)

    targets = torch.cat([torch.arange(batch_size, 2*batch_size),
                         torch.arange(0, batch_size)]).to(z.device)

    loss = F.cross_entropy(sim, targets)
    return loss

def triplet_loss(anchor, positive, negative, margin=0.2):
    pos_dist = F.l1_loss(anchor, positive, reduction='none').mean(1)
    neg_dist = F.l1_loss(anchor, negative, reduction='none').mean(1)
    return F.relu(pos_dist - neg_dist + margin).mean()

def info_nce_loss(z_anchor, z_positive, temperature=0.5):
    batch_size = z_anchor.shape[0]
    z_anchor = F.normalize(z_anchor, dim=1)
    z_positive = F.normalize(z_positive, dim=1)
    logits = torch.mm(z_anchor, z_positive.t()) / temperature
    labels = torch.arange(batch_size).to(z_anchor.device)
    return F.cross_entropy(logits, labels)

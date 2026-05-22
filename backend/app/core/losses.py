# app/core/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Operates on L2-normalized feature vectors. For each anchor, pulls
    same-class features closer and pushes different-class features apart
    via the InfoNCE objective with a temperature scaling.

    Supports hard negative mining via pair_weights: a dict mapping
    (class_i, class_j) tuples to multiplicative weights applied to the
    loss contribution of those class pairs. This forces the model to
    work harder on separating difficult pairs (e.g., visually similar species).
    """

    def __init__(self, temperature: float = 0.07, pair_weights: dict | None = None):
        super().__init__()
        self.temperature = temperature
        self.pair_weights = pair_weights or {}

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        sim = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # Hard negative mining: weight specific class pairs higher
        if self.pair_weights:
            weight_mask = torch.ones_like(mask)
            for (c1, c2), w in self.pair_weights.items():
                c1_mask = (labels == c1).float()
                c2_mask = (labels == c2).float()
                hw = c1_mask @ c2_mask.T + c2_mask @ c1_mask.T
                weight_mask = weight_mask * (1.0 + (w - 1.0) * hw)
            sim = sim * weight_mask

        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        num_positives = mask.sum(dim=1)
        mean_log_prob = (mask * log_prob).sum(dim=1) / (num_positives + 1e-12)

        valid = num_positives > 0
        if not valid.any():
            return torch.tensor(0.0, requires_grad=True, device=features.device)
        loss = -mean_log_prob[valid].mean()
        return loss


class SphericalPrototypes(nn.Module):
    """Orthogonal prototypes on the unit sphere for SpHOR-style open-set.

    Each class has a prototype vector on the unit sphere. Prototypes are
    encouraged to be orthogonal via a repulse loss, creating a well-structured
    latent space that improves open-set detection.

    Args:
        num_classes: Number of known classes.
        proj_dim: Projection dimension (default 128, matches SupCon proj head).
    """

    def __init__(self, num_classes: int, proj_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.prototypes = nn.Parameter(torch.zeros(num_classes, proj_dim))
        nn.init.kaiming_normal_(self.prototypes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between features and prototypes.

        Args:
            features: (batch_size, proj_dim) — L2-normalized projection.

        Returns:
            sim: (batch_size, num_classes) cosine similarities.
        """
        prototypes = F.normalize(self.prototypes, dim=1)
        return torch.mm(features, prototypes.T)

    def repulse_loss(self, temperature: float = 0.07) -> torch.Tensor:
        """Orthogonal repulsion loss: encourages prototypes to be orthogonal.

        Minimises squared cosine similarity between all prototype pairs.
        This creates a 'spherical' space where each class occupies a distinct
        direction, improving separability for open-set detection.

        Returns:
            Scalar loss.
        """
        prots = F.normalize(self.prototypes, dim=1)
        sim = torch.mm(prots, prots.T)
        off_diag = sim - torch.eye(self.num_classes, device=sim.device)
        loss = (off_diag ** 2).sum() / (self.num_classes * (self.num_classes - 1))
        return loss

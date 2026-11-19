# app/core/losses.py
import torch
import torch.nn.functional as F
from torch import nn


class SupConLoss(nn.Module):  # type: ignore[misc]
    """Supervised Contrastive Loss (Khosla et al., 2020).

    Operates on L2-normalized feature vectors. For each anchor, pulls
    same-class features closer and pushes different-class features apart
    via the InfoNCE objective with a temperature scaling.

    Supports hard negative mining via pair_weights: a dict mapping
    (class_i, class_j) tuples to multiplicative weights applied to the
    loss contribution of those class pairs. This forces the model to
    work harder on separating difficult pairs (e.g., visually similar species).
    """

    def __init__(self, temperature: float = 0.07, pair_weights: dict[tuple[int, int], float] | None = None) -> None:
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


class SphericalPrototypes(nn.Module):  # type: ignore[misc]
    """Orthogonal prototypes on the unit sphere for SpHOR-style open-set.

    Each class owns a prototype vector on the unit sphere. Two terms act on
    them during Phase 1:

    - a von Mises-Fisher (vMF) term with **hard assignment**: every sample is
      pulled towards the prototype of its own class, so the prototypes follow
      the data instead of staying where they were initialised;
    - a repulsion term that keeps the prototypes mutually orthogonal, which
      maximises the angular coverage of the sphere and leaves room between
      cones for unknown classes.

    The prototypes live in the SupCon *projection* space (they are discarded
    with the projection head at the end of Phase 1): their job is to shape the
    backbone representation, which is what the open-set detectors then see.

    Args:
    ----
        num_classes: Number of known classes.
        proj_dim: Projection dimension (default 128, matches SupCon proj head).

    """

    def __init__(
        self,
        num_classes: int,
        proj_dim: int = 128,
        repulse_mode: str = "squared",
        repulse_threshold: float = 0.3,
    ) -> None:
        super().__init__()
        if repulse_mode not in ("squared", "hinge"):
            raise ValueError(f"repulse_mode sconosciuta: {repulse_mode}")
        self.num_classes = num_classes
        self.repulse_mode = repulse_mode
        self.repulse_threshold = repulse_threshold
        self.prototypes = nn.Parameter(torch.zeros(num_classes, proj_dim))
        nn.init.kaiming_normal_(self.prototypes)

    @torch.no_grad()
    def init_from_features(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Initialise each prototype on the class centroid of the features.

        Called at the end of the warm-up: the prototypes start from where the
        classes already are, instead of from random directions.

        Returns
        -------
            The mean cosine between prototype k and the centroid of class k
            (1.0 = perfectly centred). Classes with no sample keep their
            current prototype and do not contribute.

        """
        feats = F.normalize(features.detach().float().to(self.prototypes.device), dim=1)
        labels = labels.detach().view(-1).to(self.prototypes.device)
        updated = self.prototypes.detach().clone()
        for k in range(self.num_classes):
            mask = labels == k
            if bool(mask.any()):
                updated[k] = F.normalize(feats[mask].mean(dim=0), dim=0)
        # Anche le classi senza campioni restano sulla sfera: l'invariante e'
        # "tutti i prototipi sono unitari", non "quelli inizializzati".
        self.prototypes.data.copy_(F.normalize(updated, dim=1))
        return self.alignment(features, labels)

    @torch.no_grad()
    def alignment(self, features: torch.Tensor, labels: torch.Tensor) -> float:
        """Mean cosine between prototype k and the centroid of class k."""
        feats = F.normalize(features.detach().float().to(self.prototypes.device), dim=1)
        labels = labels.detach().view(-1).to(self.prototypes.device)
        prots = F.normalize(self.prototypes, dim=1)
        cosines = []
        for k in range(self.num_classes):
            mask = labels == k
            if bool(mask.any()):
                cosines.append(float((F.normalize(feats[mask].mean(dim=0), dim=0) * prots[k]).sum()))
        return float(sum(cosines) / len(cosines)) if cosines else 0.0

    def vmf_loss(self, features: torch.Tensor, labels: torch.Tensor, kappa: float = 10.0) -> torch.Tensor:
        """Hard-assignment vMF negative log-likelihood.

        With L2-normalised features and prototypes the von Mises-Fisher
        log-density is proportional to ``kappa * cos(f, p_k)``, so the
        supervised term is the cross-entropy of ``kappa * cosine`` against the
        true class: every sample is assigned to (and only to) its own
        prototype.

        Args:
        ----
            features: (batch_size, proj_dim) — L2-normalised projection.
            labels: (batch_size,) — class indices in ``[0, num_classes)``.
            kappa: concentration of the vMF distribution (higher = sharper).

        Returns
        -------
            Scalar loss.

        """
        logits = kappa * self.forward(features)
        return F.cross_entropy(logits, labels.view(-1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Compute similarity scores between features and prototypes.

        Args:
        ----
            features: (batch_size, proj_dim) — L2-normalized projection.

        Returns:
        -------
            sim: (batch_size, num_classes) cosine similarities.

        """
        prototypes = F.normalize(self.prototypes, dim=1)
        return torch.mm(features, prototypes.T)

    def repulse_loss(self) -> torch.Tensor:
        """Repulsion between prototypes.

        Due modalita':

        - ``squared`` (default): penalizza *tutti* i coseni al quadrato, quindi
          spinge ogni coppia verso i 90°. E' il prior del paper, ma su specie
          ultra-fini i centroidi veri sono correlati per costruzione: spingerli
          tutti a zero allontana i prototipi dai dati.
        - ``hinge``: penalizza solo le coppie che si avvicinano troppo
          (``|cos| > repulse_threshold``). Impedisce il collasso senza
          pretendere che specie simili diventino ortogonali.

        Returns
        -------
            Scalar loss.

        """
        prots = F.normalize(self.prototypes, dim=1)
        sim = torch.mm(prots, prots.T)
        off_diag = sim - torch.eye(self.num_classes, device=sim.device)
        denom = self.num_classes * (self.num_classes - 1)
        if denom <= 0:  # una sola classe: non c'e' nulla da respingere
            return torch.zeros((), device=sim.device, requires_grad=True)
        if self.repulse_mode == "hinge":
            excess = torch.clamp(off_diag.abs() - self.repulse_threshold, min=0.0)
            return (excess**2).sum() / denom
        return (off_diag**2).sum() / denom

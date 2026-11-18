# app/core/kfold/epochs.py
import copy
import gc
import logging
import os

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from app.core.models.base import ClassifierModel
from app.core.torch_metrics import MulticlassAccuracy, MulticlassF1Score

logger = logging.getLogger(__name__)


def cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def train_one_epoch(
    model: ClassifierModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    gradient_accumulation_steps: int = 1,
) -> float:
    """Train model for one epoch with optional gradient accumulation.

    Returns
    -------
        Average training loss for the epoch

    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (inputs, labels, _) in enumerate(train_loader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels) / gradient_accumulation_steps
        loss.backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * gradient_accumulation_steps * inputs.size(0)
        num_samples += inputs.size(0)

    return running_loss / max(num_samples, 1)


def validate_one_epoch(
    model: ClassifierModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float]:
    """Validate model for one epoch.

    Returns
    -------
        Tuple of (average loss, accuracy, f1 score)

    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            running_loss += loss.item()
            num_batches += 1

            accuracy_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)

    avg_loss = running_loss / max(num_batches, 1)
    accuracy = accuracy_metric.compute().item()
    f1 = f1_metric.compute().item()

    accuracy_metric.reset()
    f1_metric.reset()

    return avg_loss, accuracy, f1


def save_checkpoint(model, optimizer, scheduler, epoch, best_f1, best_val_loss, epochs_no_improve, history, path):
    """Save training checkpoint for resume (atomic: write .tmp then rename)."""
    tmp_path = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_f1": best_f1,
            "best_val_loss": best_val_loss,
            "epochs_no_improve": epochs_no_improve,
            "history": history,
        },
        tmp_path,
    )
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    """Load training checkpoint. Returns (start_epoch, best_f1, best_val_loss, epochs_no_improve, history)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return (
        ckpt["epoch"] + 1,
        ckpt["best_f1"],
        ckpt["best_val_loss"],
        ckpt["epochs_no_improve"],
        ckpt["history"],
    )


def train(
    model: ClassifierModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    num_classes: int,
    checkpoint_dir: str | None = None,
    start_epoch: int = 0,
    patience: int = 0,
    gradient_accumulation_steps: int = 1,
    resume_from: str | None = None,
    optimizer_type: str = "adam",
    weight_decay: float = 0.0,
) -> tuple[ClassifierModel, dict]:
    """Train a classifier model.

    Args:
    ----
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to use
        num_classes: Number of classes
        checkpoint_dir: Directory for intra-training checkpoints
        start_epoch: Starting epoch (for resuming training)
        patience: Early stopping patience on val_loss (0 = disabled)
        gradient_accumulation_steps: Accumulate gradients over N batches
        resume_from: Path to checkpoint file to resume from
        optimizer_type: "adam" or "adamw"
        weight_decay: Weight decay for the optimizer

    Returns:
    -------
        Tuple of (trained model, training history dict)

    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    # Resume from checkpoint
    if resume_from and os.path.exists(resume_from):
        start_epoch, best_f1, best_val_loss, epochs_no_improve, history = load_checkpoint(
            resume_from, model, optimizer, scheduler, device
        )
        best_model_wts = copy.deepcopy(model.state_dict())
        logger.info(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Ensure checkpoint dir exists
    ckpt_path = None
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(checkpoint_dir, "last_ckpt.pt")

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )
        history["train_loss"].append(train_loss)

        # Validate
        val_loss, accuracy, f1 = validate_one_epoch(model, val_loader, criterion, device, num_classes)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)
        history["val_f1"].append(f1)

        logger.info(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f} | Val F1: {f1:.4f}"
        )

        # Save best model (based on val_f1)
        if f1 >= best_f1:
            best_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"New best model with F1: {f1:.4f}")

        # Early stopping on val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save checkpoint (overwrite — only keep latest)
        if ckpt_path:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_f1, best_val_loss, epochs_no_improve, history, ckpt_path
            )

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (no val_loss improvement for {patience} epochs)")
            break

        scheduler.step(val_loss)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    logger.info(f"Training complete. Best F1: {best_f1:.4f}")

    # Clean up checkpoint file on successful completion
    if ckpt_path and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return model, history

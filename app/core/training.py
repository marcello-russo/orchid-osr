# app/core/training.py
import copy
import logging
import numpy as np
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from sklearn.model_selection import StratifiedKFold

from app.core.models.base import ClassifierModel, ModelConfig

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: ClassifierModel,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler: Optional["torch.cuda.amp.GradScaler"] = None,
) -> float:
    """
    Train model for one epoch with optional AMP.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        scaler: GradScaler for mixed precision (None = no AMP)

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    use_amp = scaler is not None

    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def validate_one_epoch(
    model: ClassifierModel,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, float]:
    """
    Validate model for one epoch.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        num_classes: Number of classes

    Returns:
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

    return avg_loss, accuracy, f1


def train(
    model: ClassifierModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    num_classes: int,
    checkpoint_dir: Optional[str] = None,
    start_epoch: int = 0,
    patience: int = 0,
) -> tuple[ClassifierModel, dict]:
    """
    Train a classifier model.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        device: Device to use
        num_classes: Number of classes
        checkpoint_dir: Optional directory to save checkpoints
        start_epoch: Starting epoch (for resuming training)
        patience: Early stopping patience on val_loss (0 = disabled)

    Returns:
        Tuple of (trained model, training history dict)
    """
    model = model.to(device)

    # AMP for CUDA
    scaler = None
    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        torch.backends.cudnn.benchmark = True
        logger.info("AMP mixed precision enabled, cudnn.benchmark=True")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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

    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        history["train_loss"].append(train_loss)

        # Validate
        val_loss, accuracy, f1 = validate_one_epoch(
            model, val_loader, criterion, device, num_classes
        )
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(accuracy)
        history["val_f1"].append(f1)

        logger.info(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {accuracy:.4f} | "
            f"Val F1: {f1:.4f}"
        )

        # Save best model (based on val_f1)
        if f1 >= best_f1:
            best_f1 = f1
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(f"New best model with F1: {f1:.4f}")

        # Early stopping on val_loss (smoother than F1)
        # ReduceLROnPlateau patience=5, early stopping patience defaults to 15 (3:1 ratio)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (no val_loss improvement for {patience} epochs)")
            break

        scheduler.step(val_loss)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    logger.info(f"Training complete. Best F1: {best_f1:.4f}")

    return model, history


def cross_validate(
    model_config: "ModelConfig",
    dataset: torch.utils.data.Dataset,
    num_folds: int,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    device: torch.device,
    num_classes: int,
    patience: int = 0,
) -> dict:
    """
    Perform k-fold cross validation.

    Creates a new model for each fold, trains it, and aggregates results.

    Args:
        model_config: Configuration for creating models
        dataset: Full dataset to split
        num_folds: Number of folds (k)
        num_epochs: Epochs per fold
        learning_rate: Learning rate
        batch_size: Batch size
        device: Device to use
        num_classes: Number of classes

    Returns:
        Dict with fold_results, mean_metrics, and std_metrics
    """
    import numpy as np

    # Extract labels for stratified split
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label)
    labels = np.array(labels)

    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold_idx + 1}/{num_folds}")
        logger.info(f"{'='*50}")

        # Create new model for this fold using config's factory method
        model = model_config.create_model()

        # Create subset dataloaders
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        # Train
        model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            num_classes=num_classes,
            patience=patience,
        )

        # Get best metrics for this fold
        best_f1 = max(history["val_f1"])
        best_acc = max(history["val_accuracy"])

        fold_results.append({
            "fold": fold_idx + 1,
            "best_f1": best_f1,
            "best_accuracy": best_acc,
            "history": history,
        })

        logger.info(f"Fold {fold_idx + 1} - Best F1: {best_f1:.4f}, Best Acc: {best_acc:.4f}")

    # Calculate mean and std
    all_f1 = [r["best_f1"] for r in fold_results]
    all_acc = [r["best_accuracy"] for r in fold_results]

    summary = {
        "fold_results": fold_results,
        "mean_f1": np.mean(all_f1),
        "std_f1": np.std(all_f1),
        "mean_accuracy": np.mean(all_acc),
        "std_accuracy": np.std(all_acc),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"CROSS-VALIDATION RESULTS ({num_folds} folds)")
    logger.info(f"{'='*50}")
    logger.info(f"F1 Score: {summary['mean_f1']:.4f} ± {summary['std_f1']:.4f}")
    logger.info(f"Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")

    return summary

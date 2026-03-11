import torch
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import logging
import sys
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from dotenv import dotenv_values
from app.preprocessing_tools.dataset_tool import getDatasetFromFile
from app.preprocessing_tools.reproducibility import set_seed, seed_worker
import numpy as np
import random
import os
import copy
from collections import Counter

config = dotenv_values(".env") # Non più necessario, veniva usato in precedenza quando i parametri venivano passati tramite file .env e non all'interno del notebook
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_class_weights(dataset, num_classes, device):
    """Calcola i class weights basati sulla distribuzione delle classi nel dataset."""
    # Conta le occorrenze di ogni classe
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)

    # Calcola i pesi usando la formula: weight[i] = total_samples / (num_classes * count[i])
    total_samples = len(dataset)
    weights = []
    for i in range(num_classes):
        count = class_counts.get(i, 1)  # Default a 1 se la classe non è presente
        weight = total_samples / (num_classes * count)
        weights.append(weight)

    # Normalizza i pesi così che la media sia 1
    weights = np.array(weights)
    weights = weights / weights.mean()

    logging.info(f'Class counts: {dict(sorted(class_counts.items()))}')
    logging.info(f'Class weights: {weights}')

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs, classSize, modelPath, checkpoint_dir=None, start_epoch=0):
    lastF1Score = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        running_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        _, actualF1Score, _, _, val_loss = validate_one_epoch(model, val_loader, criterion, device, classSize)

        if actualF1Score >= lastF1Score or epoch < 5:
            lastF1Score = actualF1Score
            best_model_wts = copy.deepcopy(model.state_dict())
            logging.info(f'Saving best model with F1 Score: {actualF1Score}')

        scheduler.step(val_loss)

        if checkpoint_dir and (epoch + 1) % 5 == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': lastF1Score,
                'best_model_wts': best_model_wts,
            }, checkpoint_path)
            logging.info(f'Checkpoint saved at epoch {epoch+1}')

    saveModel(model, modelPath.replace('.pt', '_last.pt'))
    model.load_state_dict(best_model_wts)
    saveModel(model, modelPath)

    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss

def validate_one_epoch(model, val_loader, criterion, device, classSize):
    model.eval()
    val_loss = 0.0
    accuracy_metric = MulticlassAccuracy(num_classes=classSize, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=classSize, average='macro').to(device)
    precision_metric = MulticlassPrecision(num_classes=classSize, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=classSize, average='macro').to(device)
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            accuracy_metric.update(predicted, labels)
            f1_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
    val_loss /= len(val_loader)
    val_accuracy = accuracy_metric.compute() * 100
    f1_score = f1_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    logging.info(f'Validation Accuracy: {val_accuracy}%')
    logging.info(f'Validation F1 Score: {f1_score}')
    logging.info(f'Validation Precision: {precision}')
    logging.info(f'Validation Recall: {recall}')
    logging.info(f'Validation Loss: {val_loss}')
    return val_accuracy, f1_score, precision, recall, val_loss


def loadAndTrain(device, classSize, processedDataTrainPath, processedDataValidationPath, batchSize, epoch, numWorker, learningRate, pretrained, modelPath, seed, use_class_weights=False, checkpoint_dir=None, start_epoch=0):
    set_seed(seed)
    train_dataset = getDatasetFromFile(processedDataTrainPath)
    val_dataset = getDatasetFromFile(processedDataValidationPath)

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(val_dataset)}')

    train_loader = DataLoader(train_dataset, batchSize, shuffle=True, num_workers=numWorker, prefetch_factor=2, persistent_workers=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batchSize, shuffle=False, num_workers=numWorker, prefetch_factor=2, persistent_workers=True, worker_init_fn=seed_worker)

    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
    else:
        model = models.resnet18().to(device)

    model.fc = nn.Linear(model.fc.in_features, classSize)
    model = model.to(device)
    print(model)

    # Calcola i class weights se richiesto
    if use_class_weights:
        class_weights = compute_class_weights(train_dataset, classSize, device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
        logging.info('Using class weights in CrossEntropyLoss')
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
        logging.info('Using standard CrossEntropyLoss (no class weights)')

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    # Resume da checkpoint se necessario
    if start_epoch > 0 and checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{start_epoch}.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            best_model_wts = checkpoint.get('best_model_wts')
            logging.info(f'Resumed from epoch {start_epoch}')
        else:
            logging.warning(f'Checkpoint not found: {checkpoint_path}, starting fresh')
            start_epoch = 0

    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epoch, classSize, modelPath, checkpoint_dir, start_epoch)

    return model

def saveModel(model, modelPath):
    if not os.path.exists(os.path.dirname(modelPath)):
        os.makedirs(os.path.dirname(modelPath))
    torch.save({'model': model.state_dict()}, modelPath)



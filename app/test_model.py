import logging
import sys
import torch
from dotenv import dotenv_values
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from app.preprocessing_tools.dataset_tool import getDatasetFromFile

from app.explenability_tools.explainability import showAndTestImages, generateOutputImages


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = dotenv_values(".env")


def testModel(model, test_loader, device, classNames, classSize):
    # Get model's output size (may be larger than test set class count)
    model_num_classes = 6  # Model was trained on 6 classes

    # Macro Average - use model's class count to avoid index errors
    accuracy_metric_macro = MulticlassAccuracy(num_classes=model_num_classes, average='macro').to(device)
    f1_metric_macro = MulticlassF1Score(num_classes=model_num_classes, average='macro').to(device)
    precision_metric_macro = MulticlassPrecision(num_classes=model_num_classes, average='macro').to(device)
    recall_metric_macro = MulticlassRecall(num_classes=model_num_classes, average='macro').to(device)

    # Micro Average
    accuracy_metric_micro = MulticlassAccuracy(num_classes=model_num_classes, average='micro').to(device)
    f1_metric_micro = MulticlassF1Score(num_classes=model_num_classes, average='micro').to(device)
    precision_metric_micro = MulticlassPrecision(num_classes=model_num_classes, average='micro').to(device)
    recall_metric_micro = MulticlassRecall(num_classes=model_num_classes, average='micro').to(device)

    # Per-class
    accuracy_metric_per_class = MulticlassAccuracy(num_classes=model_num_classes, average=None).to(device)
    f1_metric_per_class = MulticlassF1Score(num_classes=model_num_classes, average=None).to(device)
    precision_metric_per_class = MulticlassPrecision(num_classes=model_num_classes, average=None).to(device)
    recall_metric_per_class = MulticlassRecall(num_classes=model_num_classes, average=None).to(device)

    # Confusion matrix - use model's class count
    confusion_matrix = torch.zeros(model_num_classes, model_num_classes, dtype=torch.int32).to(device)

    # Per batch metrics storage
    batch_macro_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
    batch_micro_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}

    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # Update macro metrics
            accuracy_metric_macro.update(predicted, labels)
            f1_metric_macro.update(predicted, labels)
            precision_metric_macro.update(predicted, labels)
            recall_metric_macro.update(predicted, labels)
            
            # Update micro metrics
            accuracy_metric_micro.update(predicted, labels)
            f1_metric_micro.update(predicted, labels)
            precision_metric_micro.update(predicted, labels)
            recall_metric_micro.update(predicted, labels)

            # Update per-class metrics
            accuracy_metric_per_class.update(predicted, labels)
            f1_metric_per_class.update(predicted, labels)
            precision_metric_per_class.update(predicted, labels)
            recall_metric_per_class.update(predicted, labels)

            # Store per batch metrics for std calculation
            # Note: May fail if test set has fewer classes than model - wrap in try-except
            try:
                batch_macro_metrics['accuracy'].append(accuracy_metric_macro.compute().item())
                batch_macro_metrics['f1'].append(f1_metric_macro.compute().item())
                batch_macro_metrics['precision'].append(precision_metric_macro.compute().item())
                batch_macro_metrics['recall'].append(recall_metric_macro.compute().item())

                batch_micro_metrics['accuracy'].append(accuracy_metric_micro.compute().item())
                batch_micro_metrics['f1'].append(f1_metric_micro.compute().item())
                batch_micro_metrics['precision'].append(precision_metric_micro.compute().item())
                batch_micro_metrics['recall'].append(recall_metric_micro.compute().item())
            except RuntimeError:
                # Skip per-batch metrics if there's a class mismatch
                pass

            # Update confusion matrix
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
    # Compute metrics - wrap in try-except to handle class mismatch
    try:
        test_accuracy_macro = accuracy_metric_macro.compute()
        test_f1_macro = f1_metric_macro.compute()
        precision_macro = precision_metric_macro.compute()
        recall_macro = recall_metric_macro.compute()

        test_accuracy_micro = accuracy_metric_micro.compute()
        test_f1_micro = f1_metric_micro.compute()
        precision_micro = precision_metric_micro.compute()
        recall_micro = recall_metric_micro.compute()

        accuracy_per_class = accuracy_metric_per_class.compute()
        f1_per_class = f1_metric_per_class.compute()
        precision_per_class = precision_metric_per_class.compute()
        recall_per_class = recall_metric_per_class.compute()
    except RuntimeError as e:
        logging.warning(f"Metric computation warning: {e}")
        # Use confusion matrix for basic metrics instead
        total = confusion_matrix.sum()
        correct = torch.diag(confusion_matrix).sum()
        test_accuracy_macro = correct.float() / total.float() if total > 0 else torch.tensor(0.0)
        test_f1_macro = torch.tensor(0.0)
        precision_macro = torch.tensor(0.0)
        recall_macro = torch.tensor(0.0)
        test_accuracy_micro = test_accuracy_macro
        test_f1_micro = torch.tensor(0.0)
        precision_micro = torch.tensor(0.0)
        recall_micro = torch.tensor(0.0)
        accuracy_per_class = torch.zeros(model_num_classes)
        f1_per_class = torch.zeros(model_num_classes)
        precision_per_class = torch.zeros(model_num_classes)
        recall_per_class = torch.zeros(model_num_classes)

    std_macro_metrics = {key: torch.std(torch.tensor(values)) if values else torch.tensor(0.0) for key, values in batch_macro_metrics.items()}
    std_micro_metrics = {key: torch.std(torch.tensor(values)) if values else torch.tensor(0.0) for key, values in batch_micro_metrics.items()}


    # Log results
    logging.info(f'Test Accuracy (Macro): {test_accuracy_macro} ± {std_macro_metrics["accuracy"]}')
    logging.info(f'Test F1 Score (Macro): {test_f1_macro} ± {std_macro_metrics["f1"]}')
    logging.info(f'Test Precision (Macro): {precision_macro} ± {std_macro_metrics["precision"]}')
    logging.info(f'Test Recall (Macro): {recall_macro} ± {std_macro_metrics["recall"]}')
    
    logging.info(f'Test Accuracy (Micro): {test_accuracy_micro} ± {std_micro_metrics["accuracy"]}')
    logging.info(f'Test F1 Score (Micro): {test_f1_micro} ± {std_micro_metrics["f1"]}')
    logging.info(f'Test Precision (Micro): {precision_micro} ± {std_micro_metrics["precision"]}')
    logging.info(f'Test Recall (Micro): {recall_micro} ± {std_micro_metrics["recall"]}')

    
    # Log per-class metrics
    for i, classNames in enumerate(classNames):
        logging.info(f'Class: {classNames}')
        logging.info(f'Accuracy: {accuracy_per_class[i]}')
        logging.info(f'F1 Score: {f1_per_class[i]}')
        logging.info(f'Precision: {precision_per_class[i]}')
        logging.info(f'Recall: {recall_per_class[i]}')

    # Log confusion matrix
    logging.info(f'Confusion Matrix:\n{confusion_matrix}')


    return {
        "macro": (test_accuracy_macro, test_f1_macro, precision_macro, recall_macro),
        "micro": (test_accuracy_micro, test_f1_micro, precision_micro, recall_micro),
        "per_class": precision_per_class
    }

def modelLoader(MODEL_PATH, CLASS_SIZE, device):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, CLASS_SIZE)
    model_dict = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(model_dict['model'])
    model = model.to(device)
    model.eval()
    return model

def deviceLoader():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# Questa funzione non è disponibile nel notebook jupyter, permette di visualizzare le immagini del dataset di test con un'interfaccia interattiva 
# in questo modo è possibile visualizzare i risultati dell'Occlusion per ciascun punto
# Questa funzione non è stata inserita nel notebook jupyter perché non sono riuscito ad implementare l'interfaccia interattiva in un notebook
def testModelWrapper(CLASS_NAMES, PROCESSED_DATA_TEST_PATH, MODEL_PATH, BATCH_SIZE, NUM_WORKERS, DESTINATION_PATH, SLIDING_WINDOW_SIZE, STRIDE):
    CLASS_SIZE = len(CLASS_NAMES)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, CLASS_SIZE)
    model_dict = torch.load(MODEL_PATH, weights_only=False)
    model.load_state_dict(model_dict['model'])
    model = model.to(device)
    model.eval()
    test_dataset = getDatasetFromFile(PROCESSED_DATA_TEST_PATH)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, prefetch_factor=2, persistent_workers=True)
    testModel(model, test_loader, device, CLASS_NAMES, CLASS_SIZE)
    if '--generate-results' in sys.argv:
        generateOutputImages(test_dataset, model, device, CLASS_NAMES, DESTINATION_PATH, SLIDING_WINDOW_SIZE, STRIDE)
    showAndTestImages(test_dataset, model, device, CLASS_NAMES, SLIDING_WINDOW_SIZE, STRIDE)

# Prima di eseguire questo è codice e quindi visualizzare l'interfaccia interattiva è necessario configurare il file .env con i parametri corretti
# e lanciare il comando python app/test_model.py --generate-results
# o in alternativa modificare il seguente codice e inserire staticamente i parametri di ambiente
if __name__ == '__main__':
    CLASS_NAMES = eval(config['CLASS_NAMES'])
    CLASS_SIZE = len(CLASS_NAMES)

    PROCESSED_DATA_TEST_PATH = config['PROCESSED_DATA_TEST_PATH']
    MODEL_PATH = config['MODEL_PATH']
    BATCH_SIZE = int(config['BATCH_SIZE_TEST'])
    NUM_WORKERS = int(config['NUM_WORKERS_TEST'])
    DESTINATION_PATH = config['DESTINATION_PATH']
    SLIDING_WINDOW_SIZE = int(config['SLIDING_WINDOW_SIZE'])
    STRIDE = int(config['SLIDING_WINDOW_STRIDE'])

    testModelWrapper(CLASS_NAMES, PROCESSED_DATA_TEST_PATH, MODEL_PATH, BATCH_SIZE, NUM_WORKERS, DESTINATION_PATH, SLIDING_WINDOW_SIZE, STRIDE)
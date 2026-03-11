#!/usr/bin/env python3
"""
Orchid Classification Pipeline - CLI Script

Esegue la pipeline completa o singole fasi.
Più robusto del notebook per esecuzioni lunghe.

Usage:
    python run_pipeline.py --all                    # Esegue tutto
    python run_pipeline.py --augment --train        # Solo augmentation + training
    python run_pipeline.py --train --epochs 50      # Training con 50 epochs
    python run_pipeline.py --test                   # Solo testing
    nohup python run_pipeline.py --all > log.txt &  # In background

Author: Davide Di Pierro (original), Marcello Russo (CLI adaptation)
"""

import argparse
import os
import random
import shutil
import sys
import torch
import logging
from datetime import datetime
from torch.utils.data import DataLoader

# Add app to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.timing import Timer
from app.preprocessing_tools.reproducibility import set_seed


# ============================================
# LOGGING SETUP
# ============================================
def setup_logging(log_dir='logs'):
    """Setup logging with timestamped file"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_formatter = logging.Formatter('%(message)s')

    # File handler - detailed
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler - concise
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f'Log file: {log_file}')

    return log_file


class MetricsLogger:
    """Logger per metriche strutturate"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.metrics = {}

    def log_phase(self, phase_name, duration=None):
        """Log phase start/end"""
        msg = f'=== {phase_name} ==='
        if duration:
            msg += f' [Duration: {duration:.2f}s]'
        logging.info(msg)

    def log_config(self, config):
        """Log configuration"""
        logging.info('--- Configuration ---')
        for key in ['seed', 'epochs', 'batch_size_train', 'learning_rate', 'use_class_weights', 'pretrained']:
            if key in config:
                logging.info(f'{key}: {config[key]}')
        logging.info('---------------------')

    def log_metrics(self, phase, metrics_dict):
        """Log metrics for a phase"""
        logging.info(f'--- {phase} Metrics ---')
        for key, value in metrics_dict.items():
            if isinstance(value, float):
                logging.info(f'{key}: {value:.4f}')
            else:
                logging.info(f'{key}: {value}')
        logging.info('----------------------')

        # Store for later
        if phase not in self.metrics:
            self.metrics[phase] = []
        self.metrics[phase].append(metrics_dict)

    def log_epoch(self, epoch, total_epochs, train_loss, val_loss, val_f1, val_acc):
        """Log epoch results"""
        msg = f'Epoch {epoch}/{total_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.2f}%'
        logging.info(msg)

    def log_class_weights(self, weights):
        """Log class weights"""
        logging.info(f'Class weights: {[round(w, 3) for w in weights.tolist()]}')

    def save_summary(self):
        """Save final summary to log"""
        logging.info('\n' + '='*60)
        logging.info('PIPELINE SUMMARY')
        logging.info('='*60)
        for phase, metrics_list in self.metrics.items():
            logging.info(f'\n{phase}:')
            for metrics in metrics_list:
                for k, v in metrics.items():
                    if isinstance(v, float):
                        logging.info(f'  {k}: {v:.4f}')
                    else:
                        logging.info(f'  {k}: {v}')


# ============================================
# DEFAULT CONFIGURATION
# ============================================
DEFAULT_CONFIG = {
    # Paths
    'raw_data_path': 'dataset/raw/',
    'augmented_data_path': 'dataset/augmented/',
    'processed_train_path': 'dataset/processed/processed_data_train.pt',
    'processed_validation_path': 'dataset/processed/processed_data_validation.pt',
    'processed_test_path': 'dataset/processed/processed_data_test.pt',
    'model_path': 'models/model.pt',
    'results_path': 'results/',

    # Classes
    'class_names': [
        'O. exaltata',
        'O. garganica',
        'O. incubacea',
        'O. majellensis',
        'O. sphegodes',
        'O. sphegodes_Palena'
    ],

    # Preprocessing
    'width': 256,
    'height': 512,
    'train_size': 0.7,
    'validation_size': 0.1,
    'batch_size_preprocess': 32,
    'num_workers_preprocess': 4,  # Ridotto per compatibilità MPS
    'should_normalize': True,

    # Training
    'batch_size_train': 8,
    'num_workers_train': 2,  # Ridotto per compatibilità MPS
    'epochs': 26,
    'learning_rate': 0.001,
    'pretrained': True,
    'use_class_weights': False,  # Disabilitato: usare augmentation bilanciata invece
    'force_cpu': False,

    # Testing
    'batch_size_test': 32,
    'num_workers_test': 2,  # Ridotto per compatibilità MPS
    'sliding_window_size': 30,
    'sliding_window_stride': 15,

    # Reproducibility
    'seed': 57,
}


def get_device(force_cpu=False):
    """Select device: MPS > CUDA > CPU"""
    if force_cpu:
        return torch.device('cpu')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def setup_reproducibility(seed):
    """Setup reproducibility settings"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    os.environ["PYTHONHASHSEED"] = "0"
    torch.use_deterministic_algorithms(True)
    set_seed(seed)


def run_augmentation(config, metrics_logger=None, force_clean=False):
    from app.preprocess_data import DynamicAugmentationWrapper
    from app.preprocessing_tools import rotation

    logging.info(f"DEBUG: force_clean={force_clean}")
    logging.info("\n" + "="*50)
    logging.info("AUGMENTATION")
    logging.info("="*50)

    augmented_path = config['augmented_data_path']
    train_path = os.path.join(augmented_path, 'train')

    # Check if augmentation already completed (all classes have ~280 images)
    if os.path.exists(train_path) and not force_clean:
        all_complete = True
        for class_name in config['class_names']:
            class_dir = os.path.join(train_path, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                if count < 250:  # Allow some tolerance
                    all_complete = False
                    break
            else:
                all_complete = False
                break

        if all_complete:
            logging.info('✓ Augmentation already completed, skipping')
            return

    # Clean output directory only if explicitly requested
    if force_clean and os.path.exists(augmented_path):
        shutil.rmtree(augmented_path)
        logging.info(f'Cleaned directory: {augmented_path}')

    work_dir = 'work'
    augmentations = [
        rotation.identity,
        rotation.horizontal_and_vertical_flip,
        rotation.horizontal_flip,
        rotation.vertical_flip
    ]

    with Timer('Augmentation') as t:
        DynamicAugmentationWrapper(
            config['raw_data_path'],
            work_dir,
            config['augmented_data_path'],
            config['class_names'],
            config['train_size'],
            config['validation_size'],
            config['width'],
            config['height'],
            False,  # dynamic_augmentation
            augmentations,
            balanced=True  # Bilancia tutte le classi allo stesso numero
        )

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
            logging.info(f'Removed directory {work_dir}')

    # Log metrics
    if metrics_logger:
        # Count images per class after augmentation
        train_path = config['augmented_data_path'] + 'train/'
        class_counts = {}
        for class_name in config['class_names']:
            class_dir = os.path.join(train_path, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
                class_counts[class_name] = count
        metrics_logger.log_metrics('Augmentation', {
            'duration_sec': t.elapsed,
            **{f'train_{k}': v for k, v in class_counts.items()}
        })

    logging.info('✓ Augmentation completata')


def run_normalization(config, metrics_logger=None):
    """Phase 2.2: Calculate MEAN and STD"""
    from app.preprocess_data import fresh_normalization

    logging.info("\n" + "="*50)
    logging.info("PHASE: Normalization")
    logging.info("="*50)

    with Timer('Normalization') as t:
        mean, std = fresh_normalization(
            config['augmented_data_path'] + 'train/',
            config['width'],
            config['height'],
            config['batch_size_preprocess'],
            config['num_workers_preprocess'],
            config['seed']
        )

    mean_str = ' '.join([str(x) for x in mean])
    std_str = ' '.join([str(x) for x in std])

    logging.info(f'MEAN: {mean_str}')
    logging.info(f'STD: {std_str}')
    logging.info('⚠ Aggiorna i valori nel config se diversi!')

    if metrics_logger:
        metrics_logger.log_metrics('Normalization', {
            'duration_sec': t.elapsed,
            'mean_R': mean[0], 'mean_G': mean[1], 'mean_B': mean[2],
            'std_R': std[0], 'std_G': std[1], 'std_B': std[2]
        })

    return mean_str, std_str


def run_preprocessing(config, mean, std, metrics_logger=None):
    """Phase 2.3: Convert images to tensors"""
    from app.preprocess_data import preprocessData

    logging.info("\n" + "="*50)
    logging.info("PHASE: Preprocessing (tensor conversion)")
    logging.info("="*50)

    paths = [
        config['augmented_data_path'] + 'train/',
        config['augmented_data_path'] + 'validation/',
        config['augmented_data_path'] + 'test/'
    ]

    with Timer('Preprocessing') as t:
        logging.info('Processing training dataset...')
        preprocessData(
            paths[0], config['processed_train_path'],
            config['width'], config['height'],
            config['batch_size_preprocess'],
            config['num_workers_preprocess'],
            config['should_normalize'], config['seed'],
            mean, std
        )

        logging.info('Processing validation dataset...')
        preprocessData(
            paths[1], config['processed_validation_path'],
            config['width'], config['height'],
            config['batch_size_preprocess'],
            config['num_workers_preprocess'],
            config['should_normalize'], config['seed'],
            mean, std
        )

        logging.info('Processing test dataset...')
        preprocessData(
            paths[2], config['processed_test_path'],
            config['width'], config['height'],
            config['batch_size_preprocess'],
            config['num_workers_preprocess'],
            config['should_normalize'], config['seed'],
            mean, std
        )

    if metrics_logger:
        metrics_logger.log_metrics('Preprocessing', {
            'duration_sec': t.elapsed
        })

    logging.info('✓ Preprocessing completato')


def run_training(config, metrics_logger=None, checkpoint_dir=None, start_epoch=0):
    """Phase 3: Train the model"""
    from app.train_model import loadAndTrain, saveModel

    logging.info("\n" + "="*50)
    logging.info("PHASE: Training")
    logging.info("="*50)

    device = get_device(config['force_cpu'])

    if device == torch.device('cpu'):
        logging.warning('⚠ WARNING: Running on CPU. Training will be slow.')

    logging.info(f'Device: {device}')
    logging.info(f'Class weights: {config["use_class_weights"]}')
    logging.info(f'Epochs: {config["epochs"]}')
    logging.info(f'Batch size: {config["batch_size_train"]}')
    logging.info(f'Learning rate: {config["learning_rate"]}')

    if start_epoch > 0:
        logging.info(f'Resuming from epoch {start_epoch}')

    with Timer('Training') as t:
        model = loadAndTrain(
            device,
            len(config['class_names']),
            config['processed_train_path'],
            config['processed_validation_path'],
            config['batch_size_train'],
            config['epochs'],
            config['num_workers_train'],
            config['learning_rate'],
            config['pretrained'],
            config['model_path'],
            config['seed'],
            config['use_class_weights'],
            checkpoint_dir=checkpoint_dir,
            start_epoch=start_epoch
        )

    saveModel(model, config['model_path'])

    if metrics_logger:
        metrics_logger.log_metrics('Training', {
            'duration_sec': t.elapsed,
            'epochs': config['epochs'],
            'device': str(device),
            'class_weights': config['use_class_weights']
        })

    logging.info('✓ Training completato')

    return model


def run_testing(config, metrics_logger=None):
    """Phase 4: Test the model"""
    from app.test_model import deviceLoader, modelLoader, testModel, generateOutputImages
    from app.preprocessing_tools.dataset_tool import getDatasetFromFile

    logging.info("\n" + "="*50)
    logging.info("PHASE: Testing")
    logging.info("="*50)

    device = deviceLoader()
    model = modelLoader(config['model_path'], len(config['class_names']), device)

    testDataset = getDatasetFromFile(config['processed_test_path'])
    testDataLoader = DataLoader(
        testDataset,
        config['batch_size_test'],
        shuffle=True,
        num_workers=config['num_workers_test']
    )

    with Timer('Testing') as t:
        # testModel prints metrics internally, we capture them via logging
        testModel(model, testDataLoader, device, config['class_names'], len(config['class_names']))

        generateOutputImages(
            testDataset, model, device,
            config['class_names'], config['results_path'],
            config['sliding_window_size'],
            config['sliding_window_stride']
        )

    if metrics_logger:
        metrics_logger.log_metrics('Testing', {
            'duration_sec': t.elapsed,
            'test_samples': len(testDataset)
        })

    logging.info('✓ Testing completato')


def run_all(config, mean, std, metrics_logger=None):
    """Run complete pipeline"""
    logging.info("\n" + "="*60)
    logging.info("RUNNING COMPLETE PIPELINE")
    logging.info("="*60)

    with Timer('Complete Pipeline') as t:
        run_augmentation(config, metrics_logger)
        run_normalization(config, metrics_logger)
        run_preprocessing(config, mean, std, metrics_logger)
        run_training(config, metrics_logger)
        run_testing(config, metrics_logger)

    logging.info("\n" + "="*60)
    logging.info("✓ PIPELINE COMPLETATA")
    logging.info(f"Total duration: {t.elapsed:.2f}s")
    logging.info("="*60)

    if metrics_logger:
        metrics_logger.log_metrics('Pipeline', {'total_duration_sec': t.elapsed})
        metrics_logger.save_summary()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Orchid Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --all                    # Run complete pipeline
  python run_pipeline.py --augment                # Only augmentation
  python run_pipeline.py --train --epochs 50      # Training with 50 epochs
  python run_pipeline.py --train --test           # Training + testing
  nohup python run_pipeline.py --all > log.txt &  # Run in background
        """
    )

    # Phase selectors
    phase_group = parser.add_argument_group('Phases (select one or more)')
    phase_group.add_argument('--all', action='store_true', help='Run all phases')
    phase_group.add_argument('--augment', action='store_true', help='Run augmentation')
    phase_group.add_argument('--normalize', action='store_true', help='Calculate mean/std')
    phase_group.add_argument('--preprocess', action='store_true', help='Convert to tensors')
    phase_group.add_argument('--train', action='store_true', help='Train model')
    phase_group.add_argument('--test', action='store_true', help='Test model')

    # Training parameters
    train_group = parser.add_argument_group('Training Parameters')
    train_group.add_argument('--epochs', type=int, default=DEFAULT_CONFIG['epochs'],
                            help=f"Number of epochs (default: {DEFAULT_CONFIG['epochs']})")
    train_group.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size_train'],
                            help=f"Batch size (default: {DEFAULT_CONFIG['batch_size_train']})")
    train_group.add_argument('--lr', type=float, default=DEFAULT_CONFIG['learning_rate'],
                            help=f"Learning rate (default: {DEFAULT_CONFIG['learning_rate']})")
    train_group.add_argument('--no-class-weights', action='store_true',
                            help='Disable class weights in loss function')
    train_group.add_argument('--force-cpu', action='store_true',
                            help='Force CPU (disable GPU)')

    # Other parameters
    other_group = parser.add_argument_group('Other Parameters')
    other_group.add_argument('--seed', type=int, default=DEFAULT_CONFIG['seed'],
                            help=f"Random seed (default: {DEFAULT_CONFIG['seed']})")
    other_group.add_argument('--mean', type=str,
                            default='0.538217306137085 0.5433580875396729 0.368904173374176',
                            help='Mean for normalization (space-separated)')
    other_group.add_argument('--std', type=str,
                            default='0.20864470303058624 0.23085597157478333 0.2029736042022705',
                            help='Std for normalization (space-separated)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    log_file = setup_logging()
    metrics_logger = MetricsLogger(log_file)

    # Build config from defaults and args
    config = DEFAULT_CONFIG.copy()
    config['epochs'] = args.epochs
    config['batch_size_train'] = args.batch_size
    config['learning_rate'] = args.lr
    config['use_class_weights'] = not args.no_class_weights
    config['force_cpu'] = args.force_cpu
    config['seed'] = args.seed

    # Setup reproducibility
    setup_reproducibility(config['seed'])

    # Print config
    logging.info("="*60)
    logging.info("ORCHID CLASSIFICATION PIPELINE")
    logging.info("="*60)
    metrics_logger.log_config(config)

    # Determine which phases to run
    if args.all:
        run_all(config, args.mean, args.std, metrics_logger)
    else:
        if not any([args.augment, args.normalize, args.preprocess, args.train, args.test]):
            logging.error("Errore: specifica almeno una fase (--all, --augment, --normalize, --preprocess, --train, --test)")
            sys.exit(1)

        if args.augment:
            with Timer('Augmentation'):
                run_augmentation(config, metrics_logger)

        if args.normalize:
            with Timer('Normalization'):
                run_normalization(config, metrics_logger)

        if args.preprocess:
            with Timer('Preprocessing'):
                run_preprocessing(config, args.mean, args.std, metrics_logger)

        if args.train:
            with Timer('Training'):
                run_training(config, metrics_logger)

        if args.test:
            with Timer('Testing'):
                run_testing(config, metrics_logger)

        metrics_logger.save_summary()

    logging.info("\n✓ Done!")


if __name__ == '__main__':
    main()

"""
Training pipeline for MMD-based domain adaptation.

This is an alternative to DANN that uses Maximum Mean Discrepancy
for domain adaptation instead of adversarial training.

Usage:
    python train_mmd.py [--config path/to/config.json]
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import VADataPreprocessor
from src.models.mmd_model import MMDModel
from src.validation import EmbeddingValidator, TransferValidator
from src.utils import (
    set_seed, CheckpointManager, EarlyStopping, ExperimentLogger,
    count_parameters, print_metrics, get_device, AverageMeter
)


class MMDTrainer:
    """Trainer for MMD-based domain adaptation."""

    def __init__(self, config):
        """
        Args:
            config: Dictionary of training configuration
        """
        self.config = config
        set_seed(config['seed'])

        # Setup device
        self.device = get_device(prefer_cuda=config.get('use_cuda', True))

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.checkpoint_mgr = None
        self.early_stopping = None
        self.logger = None

        # Class names
        self.class_names = [
            'diarrhea', 'hiv', 'injury', 'malaria',
            'other_ch', 'other_infections', 'pneumonia'
        ]

    def setup_data(self):
        """Load and preprocess data."""
        print("="*80)
        print("SETTING UP DATA")
        print("="*80)

        # Initialize preprocessor
        self.preprocessor = VADataPreprocessor(
            iv5_path=self.config['iv5_path'],
            phmrc_path=self.config['phmrc_path']
        )

        # Load and preprocess
        self.data = self.preprocessor.load_and_preprocess()

        # Create splits
        self.splits = self.preprocessor.create_splits(
            self.data,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['seed']
        )

        # Create dataloaders
        self.dataloaders = self.preprocessor.create_dataloaders(
            self.splits,
            batch_size=self.config['batch_size'],
            use_balanced_sampling=self.config['use_balanced_sampling']
        )

        # Get feature dimensions
        self.feature_dims = self.preprocessor.get_feature_dims()

        print("\nData setup complete!")

    def setup_model(self):
        """Initialize model, optimizer, and training components."""
        print("\n" + "="*80)
        print("SETTING UP MMD MODEL")
        print("="*80)

        # Create model
        self.model = MMDModel(
            iv5_dim=self.feature_dims['iv5'],
            phmrc_dim=self.feature_dims['phmrc'],
            num_classes=len(self.class_names),
            encoder_hidden=self.config['encoder_hidden'],
            encoder_output=self.config['encoder_output'],
            embedding_dim=self.config['embedding_dim'],
            dropout=self.config['dropout'],
            kernel_sigmas=self.config.get('kernel_sigmas', None)
        ).to(self.device)

        # Print model info
        param_counts = count_parameters(self.model)
        print(f"Model created with {param_counts['total_str']} parameters")
        print(f"  Trainable: {param_counts['trainable_str']}")

        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )

        # Setup learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['num_epochs'],
            eta_min=self.config['learning_rate'] * 0.01
        )

        # Setup checkpoint manager
        self.checkpoint_mgr = CheckpointManager(
            save_dir=os.path.join(self.config['save_dir'], 'checkpoints_mmd'),
            mode='max'
        )

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            mode='max',
            min_delta=0.001
        )

        # Setup experiment logger
        self.logger = ExperimentLogger(
            log_dir=self.config['save_dir'],
            experiment_name=self.config.get('experiment_name', 'mmd_experiment')
        )
        self.logger.log_config(self.config)

        print("Model setup complete!")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()

        # Loss functions
        classification_loss_fn = nn.CrossEntropyLoss()

        # Metrics
        class_loss_meter = AverageMeter()
        mmd_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        class_acc_meter = AverageMeter()

        # Get iterators
        iv5_iter = iter(self.dataloaders['iv5']['train'])
        phmrc_iter = iter(self.dataloaders['phmrc']['train'])

        num_batches = min(
            len(self.dataloaders['iv5']['train']),
            len(self.dataloaders['phmrc']['train'])
        )

        for batch_idx in range(num_batches):
            # Get batches
            try:
                iv5_batch = next(iv5_iter)
            except StopIteration:
                iv5_iter = iter(self.dataloaders['iv5']['train'])
                iv5_batch = next(iv5_iter)

            try:
                phmrc_batch = next(phmrc_iter)
            except StopIteration:
                phmrc_iter = iter(self.dataloaders['phmrc']['train'])
                phmrc_batch = next(phmrc_iter)

            # Move to device
            iv5_features = iv5_batch['features'].to(self.device)
            iv5_labels = iv5_batch['label'].to(self.device)

            phmrc_features = phmrc_batch['features'].to(self.device)
            phmrc_labels = phmrc_batch['label'].to(self.device)

            # Forward pass
            output = self.model(
                iv5_features=iv5_features,
                phmrc_features=phmrc_features,
                compute_mmd=True
            )

            # Combine labels
            all_labels = torch.cat([iv5_labels, phmrc_labels])

            # Compute losses
            class_loss = classification_loss_fn(output['class_logits'], all_labels)
            mmd_loss = output['mmd_loss']

            # Total loss: classification + MMD
            total_loss = (
                self.config['lambda_cls'] * class_loss +
                self.config['lambda_mmd'] * mmd_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Compute accuracy
            class_preds = torch.argmax(output['class_logits'], dim=1)
            class_acc = (class_preds == all_labels).float().mean()

            # Update meters
            batch_size = len(iv5_features) + len(phmrc_features)
            class_loss_meter.update(class_loss.item(), batch_size)
            mmd_loss_meter.update(mmd_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), batch_size)
            class_acc_meter.update(class_acc.item(), batch_size)

            # Print progress
            if (batch_idx + 1) % self.config.get('print_freq', 10) == 0:
                print(f"  Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {total_loss_meter.avg:.4f} "
                      f"(cls: {class_loss_meter.avg:.4f}, mmd: {mmd_loss_meter.avg:.4f}) "
                      f"Acc: {class_acc_meter.avg:.4f}")

        return {
            'class_loss': class_loss_meter.avg,
            'mmd_loss': mmd_loss_meter.avg,
            'total_loss': total_loss_meter.avg,
            'class_acc': class_acc_meter.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def validate_epoch(self):
        """Validate on validation sets."""
        self.model.eval()

        metrics = {}

        for dataset_name in ['iv5', 'phmrc']:
            dataloader = self.dataloaders[dataset_name]['val']

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in dataloader:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].cpu().numpy()

                    # Make predictions
                    preds = self.model.predict(features, dataset=dataset_name)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels)

            # Compute accuracy
            import numpy as np
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            accuracy = (all_preds == all_labels).mean()

            metrics[f'{dataset_name}_val_acc'] = accuracy

        # Average validation accuracy
        metrics['avg_val_acc'] = (metrics['iv5_val_acc'] + metrics['phmrc_val_acc']) / 2

        return metrics

    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("STARTING TRAINING (MMD)")
        print("="*80)

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            print("-" * 80)

            # Train
            train_metrics = self.train_epoch(epoch)
            print("\nTraining metrics:")
            print_metrics(train_metrics, prefix='  ')

            # Validate
            val_metrics = self.validate_epoch()
            print("\nValidation metrics:")
            print_metrics(val_metrics, prefix='  ')

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Log metrics
            self.logger.log_epoch(epoch, epoch_metrics)

            # Update learning rate
            self.scheduler.step()

            # Save checkpoint
            is_best = self.checkpoint_mgr.is_better(val_metrics['avg_val_acc'])
            checkpoint_path = self.checkpoint_mgr.save_checkpoint(
                self.model, self.optimizer, epoch,
                val_metrics['avg_val_acc'], is_best
            )

            if is_best:
                print(f"\n✓ New best model! Val acc: {val_metrics['avg_val_acc']:.4f}")
                print(f"  Saved to: {checkpoint_path}")

            # Early stopping
            if self.early_stopping(val_metrics['avg_val_acc']):
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)


def main():
    """Main training function."""
    # Default configuration for MMD
    config = {
        # Data
        'iv5_path': 'data/IV5_child_8categories.csv',
        'phmrc_path': 'data/PHMRC_child_8categories.csv',
        'test_size': 0.15,
        'val_size': 0.15,

        # Model architecture (same as high regularization)
        'encoder_hidden': 128,
        'encoder_output': 64,
        'embedding_dim': 32,
        'dropout': 0.5,
        'kernel_sigmas': [0.01, 0.1, 1, 10, 100],  # MMD kernel bandwidths

        # Training
        'batch_size': 32,
        'num_epochs': 150,
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'use_balanced_sampling': True,

        # Domain adaptation (MMD instead of adversarial)
        'lambda_cls': 1.0,
        'lambda_mmd': 1.0,  # MMD loss weight

        # Regularization
        'patience': 20,

        # System
        'seed': 42,
        'use_cuda': True,
        'print_freq': 10,

        # Logging
        'save_dir': 'results',
        'experiment_name': 'mmd_model_v1'
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train MMD model for VA cross-dataset classification')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    # Create trainer
    trainer = MMDTrainer(config)

    # Setup
    trainer.setup_data()
    trainer.setup_model()

    # Train
    trainer.train()

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == '__main__':
    main()

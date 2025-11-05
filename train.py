"""
Training pipeline for Domain-Adversarial Neural Network (DANN).

This script:
1. Loads and preprocesses IV5 and PHMRC datasets
2. Creates DANN model with dataset-specific encoders
3. Trains with combined classification and domain adaptation losses
4. Validates periodically and saves best model
5. Logs all metrics for analysis

Usage:
    python train.py [--config path/to/config.json]
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
from src.models.dann import DANN
from src.validation import EmbeddingValidator, TransferValidator
from src.utils import (
    set_seed, get_alpha_schedule, CheckpointManager,
    EarlyStopping, ExperimentLogger, count_parameters,
    print_metrics, get_device, AverageMeter
)


class DANNTrainer:
    """Trainer for Domain-Adversarial Neural Network."""

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
        print("SETTING UP MODEL")
        print("="*80)

        # Create model
        self.model = DANN(
            iv5_dim=self.feature_dims['iv5'],
            phmrc_dim=self.feature_dims['phmrc'],
            num_classes=len(self.class_names),
            encoder_hidden=self.config['encoder_hidden'],
            encoder_output=self.config['encoder_output'],
            embedding_dim=self.config['embedding_dim'],
            dropout=self.config['dropout'],
            lambda_domain=self.config['lambda_domain']
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
            save_dir=os.path.join(self.config['save_dir'], 'checkpoints'),
            mode='max'  # Maximize validation accuracy
        )

        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['patience'],
            mode='max',  # Maximize validation accuracy
            min_delta=0.001
        )

        # Setup experiment logger
        self.logger = ExperimentLogger(
            log_dir=self.config['save_dir'],
            experiment_name=self.config.get('experiment_name')
        )
        self.logger.log_config(self.config)

        print("Model setup complete!")

    def train_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        # Compute alpha for domain adaptation
        alpha = get_alpha_schedule(
            epoch,
            self.config['num_epochs'],
            schedule_type=self.config['alpha_schedule']
        )

        # Loss functions
        classification_loss_fn = nn.CrossEntropyLoss()
        domain_loss_fn = nn.CrossEntropyLoss()

        # Metrics
        class_loss_meter = AverageMeter()
        domain_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        class_acc_meter = AverageMeter()
        domain_acc_meter = AverageMeter()

        # Get iterators for both datasets
        iv5_iter = iter(self.dataloaders['iv5']['train'])
        phmrc_iter = iter(self.dataloaders['phmrc']['train'])

        # Determine number of batches (use the smaller dataset)
        num_batches = min(
            len(self.dataloaders['iv5']['train']),
            len(self.dataloaders['phmrc']['train'])
        )

        for batch_idx in range(num_batches):
            # Get batches from both datasets
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

            # Create domain labels
            batch_size = len(iv5_features) + len(phmrc_features)
            domain_labels = torch.cat([
                torch.zeros(len(iv5_features), dtype=torch.long),  # IV5 = 0
                torch.ones(len(phmrc_features), dtype=torch.long)  # PHMRC = 1
            ]).to(self.device)

            # Forward pass
            output = self.model(
                iv5_features=iv5_features,
                phmrc_features=phmrc_features,
                alpha=alpha
            )

            # Combine labels
            all_labels = torch.cat([iv5_labels, phmrc_labels])

            # Compute losses
            class_loss = classification_loss_fn(output['class_logits'], all_labels)
            domain_loss = domain_loss_fn(output['domain_logits'], domain_labels)

            # Total loss: classification + domain adaptation
            total_loss = (
                self.config['lambda_cls'] * class_loss +
                self.config['lambda_adv'] * domain_loss
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Compute accuracies
            class_preds = torch.argmax(output['class_logits'], dim=1)
            class_acc = (class_preds == all_labels).float().mean()

            domain_preds = torch.argmax(output['domain_logits'], dim=1)
            domain_acc = (domain_preds == domain_labels).float().mean()

            # Update meters
            class_loss_meter.update(class_loss.item(), batch_size)
            domain_loss_meter.update(domain_loss.item(), batch_size)
            total_loss_meter.update(total_loss.item(), batch_size)
            class_acc_meter.update(class_acc.item(), batch_size)
            domain_acc_meter.update(domain_acc.item(), batch_size)

            # Print progress
            if (batch_idx + 1) % self.config.get('print_freq', 10) == 0:
                print(f"  Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {total_loss_meter.avg:.4f} "
                      f"(cls: {class_loss_meter.avg:.4f}, dom: {domain_loss_meter.avg:.4f}) "
                      f"Acc: {class_acc_meter.avg:.4f} "
                      f"DomAcc: {domain_acc_meter.avg:.4f} "
                      f"Alpha: {alpha:.3f}")

        return {
            'class_loss': class_loss_meter.avg,
            'domain_loss': domain_loss_meter.avg,
            'total_loss': total_loss_meter.avg,
            'class_acc': class_acc_meter.avg,
            'domain_acc': domain_acc_meter.avg,
            'alpha': alpha,
            'lr': self.optimizer.param_groups[0]['lr']
        }

    def validate_epoch(self):
        """
        Validate on validation sets.

        Returns:
            Dictionary of validation metrics
        """
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
        print("STARTING TRAINING")
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

    def evaluate(self):
        """Final evaluation on test sets."""
        print("\n" + "="*80)
        print("FINAL EVALUATION")
        print("="*80)

        # Load best model
        print("\nLoading best model...")
        self.checkpoint_mgr.load_checkpoint(self.model)

        # Create validators
        embedding_validator = EmbeddingValidator(
            self.model, self.device, self.class_names
        )

        transfer_validator = TransferValidator(
            self.model, self.device, self.class_names
        )

        # 1. Embedding quality evaluation
        print("\n" + "="*80)
        print("EMBEDDING QUALITY EVALUATION")
        print("="*80)

        embedding_metrics = embedding_validator.evaluate_embedding_quality(
            self.dataloaders['iv5']['test'],
            self.dataloaders['phmrc']['test']
        )

        # Visualize embeddings
        print("\nGenerating embedding visualizations...")
        fig = embedding_validator.visualize_embeddings(
            [embedding_metrics['iv5_data'], embedding_metrics['phmrc_data']],
            save_path=os.path.join(self.config['save_dir'], 'embeddings_tsne.png'),
            method='tsne'
        )

        # 2. Cross-dataset transfer evaluation
        print("\n" + "="*80)
        print("CROSS-DATASET TRANSFER EVALUATION")
        print("="*80)

        # IV5 → PHMRC
        iv5_to_phmrc = transfer_validator.evaluate_transfer(
            self.dataloaders['iv5']['test'],
            self.dataloaders['phmrc']['test'],
            'IV5', 'PHMRC'
        )

        # PHMRC → IV5
        phmrc_to_iv5 = transfer_validator.evaluate_transfer(
            self.dataloaders['phmrc']['test'],
            self.dataloaders['iv5']['test'],
            'PHMRC', 'IV5'
        )

        # Plot confusion matrices
        transfer_validator.plot_confusion_matrix(
            iv5_to_phmrc['confusion_matrix'],
            save_path=os.path.join(self.config['save_dir'], 'cm_iv5_to_phmrc.png')
        )

        transfer_validator.plot_confusion_matrix(
            phmrc_to_iv5['confusion_matrix'],
            save_path=os.path.join(self.config['save_dir'], 'cm_phmrc_to_iv5.png')
        )

        # 3. Log final results
        final_results = {
            'embedding_quality': {
                'silhouette_score': embedding_metrics['silhouette_score'],
                'nn_purity': embedding_metrics['nn_purity'],
                'cross_dataset_nn_agreement': embedding_metrics['cross_dataset_nn_agreement'],
                'domain_classification_acc': embedding_metrics['domain_classification_acc']
            },
            'transfer_iv5_to_phmrc': {
                'accuracy': iv5_to_phmrc['accuracy'],
                'macro_f1': iv5_to_phmrc['macro_f1'],
                'weighted_f1': iv5_to_phmrc['weighted_f1']
            },
            'transfer_phmrc_to_iv5': {
                'accuracy': phmrc_to_iv5['accuracy'],
                'macro_f1': phmrc_to_iv5['macro_f1'],
                'weighted_f1': phmrc_to_iv5['weighted_f1']
            }
        }

        self.logger.log_final_results(final_results)

        print("\n" + "="*80)
        print("FINAL RESULTS SUMMARY")
        print("="*80)
        print("\nEmbedding Quality:")
        print_metrics(final_results['embedding_quality'], prefix='  ')
        print("\nTransfer (IV5 → PHMRC):")
        print_metrics(final_results['transfer_iv5_to_phmrc'], prefix='  ')
        print("\nTransfer (PHMRC → IV5):")
        print_metrics(final_results['transfer_phmrc_to_iv5'], prefix='  ')

        return final_results


def main():
    """Main training function."""
    # Default configuration
    config = {
        # Data
        'iv5_path': 'data/IV5_child_8categories.csv',
        'phmrc_path': 'data/PHMRC_child_8categories.csv',
        'test_size': 0.15,
        'val_size': 0.15,

        # Model architecture
        'encoder_hidden': 256,
        'encoder_output': 128,
        'embedding_dim': 64,
        'dropout': 0.3,

        # Training
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'use_balanced_sampling': True,

        # Domain adaptation
        'lambda_cls': 1.0,
        'lambda_adv': 1.0,
        'lambda_domain': 1.0,
        'alpha_schedule': 'exp',  # 'linear', 'exp', or 'const'

        # Regularization
        'patience': 15,

        # System
        'seed': 42,
        'use_cuda': True,
        'print_freq': 10,

        # Logging
        'save_dir': 'results',
        'experiment_name': None
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DANN for VA cross-dataset classification')
    parser.add_argument('--config', type=str, help='Path to config JSON file')
    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config.update(json.load(f))

    # Create trainer
    trainer = DANNTrainer(config)

    # Setup
    trainer.setup_data()
    trainer.setup_model()

    # Train
    trainer.train()

    # Evaluate
    results = trainer.evaluate()

    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)


if __name__ == '__main__':
    main()

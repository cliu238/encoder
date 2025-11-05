"""
Utility functions for training and evaluation.

Includes:
- Random seed setting
- Checkpoint management
- Alpha scheduling for domain adaptation
- Experiment logging
- Early stopping
"""

import os
import json
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_alpha_schedule(epoch, max_epochs, schedule_type='linear', p=10.0):
    """
    Compute alpha value for domain adaptation weight scheduling.

    The alpha parameter controls gradient reversal strength. Common schedules:
    - 'linear': Linearly increase from 0 to 1
    - 'exp': Exponentially increase (slow start, fast end)
    - 'const': Constant value of 1.0

    Args:
        epoch: Current epoch (0-indexed)
        max_epochs: Total number of epochs
        schedule_type: Type of schedule ('linear', 'exp', 'const')
        p: Parameter for exponential schedule (higher = slower start)

    Returns:
        Alpha value in range [0, 1]
    """
    if schedule_type == 'const':
        return 1.0
    elif schedule_type == 'linear':
        return float(epoch) / max_epochs
    elif schedule_type == 'exp':
        # Exponential schedule: 2/(1+exp(-p*progress)) - 1
        progress = float(epoch) / max_epochs
        return 2.0 / (1.0 + np.exp(-p * progress)) - 1.0
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


class CheckpointManager:
    """Manages model checkpoints and best model tracking."""

    def __init__(self, save_dir, mode='min'):
        """
        Args:
            save_dir: Directory to save checkpoints
            mode: 'min' or 'max' for best metric tracking
        """
        self.save_dir = save_dir
        self.mode = mode
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(save_dir, exist_ok=True)

    def is_better(self, metric):
        """Check if metric is better than current best."""
        if self.mode == 'min':
            return metric < self.best_metric
        else:
            return metric > self.best_metric

    def save_checkpoint(self, model, optimizer, epoch, metric, is_best=False):
        """
        Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            metric: Metric value
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metric': metric,
            'best_metric': self.best_metric
        }

        # Save latest checkpoint
        latest_path = os.path.join(self.save_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            self.best_metric = metric
            best_path = os.path.join(self.save_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            return best_path

        return latest_path

    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None):
        """
        Load model checkpoint.

        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            checkpoint_path: Path to checkpoint (default: best checkpoint)

        Returns:
            Epoch number
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'checkpoint_best.pt')

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.best_metric = checkpoint['best_metric']

        return checkpoint['epoch']


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric tracking
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False

    def __call__(self, metric):
        """
        Check if training should stop.

        Args:
            metric: Current metric value

        Returns:
            True if training should stop
        """
        if self.mode == 'min':
            improved = metric < (self.best_metric - self.min_delta)
        else:
            improved = metric > (self.best_metric + self.min_delta)

        if improved:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ExperimentLogger:
    """Logger for experiment tracking."""

    def __init__(self, log_dir, experiment_name=None):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (default: timestamp)
        """
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f'{experiment_name}.json')
        self.metrics = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': {},
            'epochs': []
        }

    def log_config(self, config):
        """Log experiment configuration."""
        self.metrics['config'] = config
        self._save()

    def log_epoch(self, epoch, metrics):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        epoch_data = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics['epochs'].append(epoch_data)
        self._save()

    def log_final_results(self, results):
        """Log final evaluation results."""
        self.metrics['final_results'] = results
        self.metrics['end_time'] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """Save metrics to JSON file."""
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_best_epoch(self, metric_name, mode='max'):
        """
        Get the epoch with the best metric value.

        Args:
            metric_name: Name of metric to track
            mode: 'min' or 'max'

        Returns:
            Epoch number and metric value
        """
        if not self.metrics['epochs']:
            return None, None

        epochs = self.metrics['epochs']
        metric_values = [e.get(metric_name, float('-inf' if mode == 'max' else 'inf'))
                        for e in epochs]

        if mode == 'max':
            best_idx = np.argmax(metric_values)
        else:
            best_idx = np.argmin(metric_values)

        best_epoch = epochs[best_idx]['epoch']
        best_value = metric_values[best_idx]

        return best_epoch, best_value


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with total and trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_str': f'{total:,}',
        'trainable_str': f'{trainable:,}'
    }


def print_metrics(metrics, prefix=''):
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for each line
    """
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f'{prefix}{key}: {value:.4f}')
        else:
            print(f'{prefix}{key}: {value}')


def create_experiment_summary(config, train_metrics, val_metrics, test_metrics=None):
    """
    Create a comprehensive experiment summary.

    Args:
        config: Experiment configuration
        train_metrics: Training metrics
        val_metrics: Validation metrics
        test_metrics: Optional test metrics

    Returns:
        Summary dictionary
    """
    summary = {
        'config': config,
        'training': train_metrics,
        'validation': val_metrics
    }

    if test_metrics is not None:
        summary['test'] = test_metrics

    summary['timestamp'] = datetime.now().isoformat()

    return summary


def get_device(prefer_cuda=True):
    """
    Get the best available device.

    Args:
        prefer_cuda: Whether to prefer CUDA over CPU

    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with new value.

        Args:
            val: New value
            n: Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")

    # Test seed setting
    print("\nTesting set_seed...")
    set_seed(42)
    r1 = torch.rand(3)
    set_seed(42)
    r2 = torch.rand(3)
    assert torch.allclose(r1, r2), "Seed setting failed"
    print("  ✓ Seed setting works")

    # Test alpha schedule
    print("\nTesting alpha schedules...")
    for schedule in ['linear', 'exp', 'const']:
        alphas = [get_alpha_schedule(e, 10, schedule) for e in range(11)]
        print(f"  {schedule}: {[f'{a:.2f}' for a in alphas]}")

    # Test checkpoint manager
    print("\nTesting CheckpointManager...")
    from src.models.dann import DANN
    model = DANN()
    optimizer = torch.optim.Adam(model.parameters())

    checkpoint_mgr = CheckpointManager('results/checkpoints_test', mode='min')
    checkpoint_mgr.save_checkpoint(model, optimizer, 0, 0.5, is_best=True)
    print("  ✓ Checkpoint saved")

    # Test early stopping
    print("\nTesting EarlyStopping...")
    early_stop = EarlyStopping(patience=3, mode='min')
    metrics = [1.0, 0.8, 0.6, 0.7, 0.8, 0.9]
    for i, m in enumerate(metrics):
        should_stop = early_stop(m)
        print(f"  Epoch {i}, metric={m:.2f}, stop={should_stop}")

    # Test experiment logger
    print("\nTesting ExperimentLogger...")
    logger = ExperimentLogger('results', 'test_exp')
    logger.log_config({'lr': 0.001, 'batch_size': 32})
    logger.log_epoch(0, {'train_loss': 1.0, 'val_acc': 0.5})
    logger.log_epoch(1, {'train_loss': 0.8, 'val_acc': 0.6})
    best_epoch, best_val = logger.get_best_epoch('val_acc', mode='max')
    print(f"  Best epoch: {best_epoch}, val_acc: {best_val}")

    # Test parameter counting
    print("\nTesting count_parameters...")
    param_counts = count_parameters(model)
    print(f"  Total params: {param_counts['total_str']}")
    print(f"  Trainable params: {param_counts['trainable_str']}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_utils()

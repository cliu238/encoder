"""
Shared Embedding Space for Verbal Autopsy Cross-Dataset Classification

This package implements Domain-Adversarial Neural Networks (DANN) for learning
unified representations across heterogeneous VA datasets.
"""

__version__ = '1.0.0'

from .data_preprocessing import VADataPreprocessor, VADataset
from .validation import EmbeddingValidator, TransferValidator
from .utils import (
    set_seed,
    get_alpha_schedule,
    CheckpointManager,
    EarlyStopping,
    ExperimentLogger,
    get_device
)

__all__ = [
    'VADataPreprocessor',
    'VADataset',
    'EmbeddingValidator',
    'TransferValidator',
    'set_seed',
    'get_alpha_schedule',
    'CheckpointManager',
    'EarlyStopping',
    'ExperimentLogger',
    'get_device'
]

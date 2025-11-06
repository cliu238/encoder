"""
Simplified evaluation script that avoids sklearn compatibility issues.

This script evaluates the trained model on test set and computes:
1. Classification accuracy on both datasets
2. Domain classification accuracy (critical metric)
3. Confusion matrices
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import VADataPreprocessor
from src.models.dann import DANN
from src.utils import set_seed, get_device


def load_model(checkpoint_path, device, config_path='results/high_regularization_v1.json'):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load config from training log
    with open(config_path, 'r') as f:
        training_log = json.load(f)
    config = training_log['config']

    # Extract architecture params
    encoder_hidden = config.get('encoder_hidden', 256)
    encoder_output = config.get('encoder_output', 128)
    embedding_dim = config.get('embedding_dim', 64)
    dropout = config.get('dropout', 0.3)

    print(f"  Architecture: hidden={encoder_hidden}, output={encoder_output}, embed={embedding_dim}")

    # We need to know input dimensions - load data to get this
    preprocessor = VADataPreprocessor(
        iv5_path=config.get('iv5_path', 'data/IV5_child_8categories.csv'),
        phmrc_path=config.get('phmrc_path', 'data/PHMRC_child_8categories.csv')
    )
    data_dict = preprocessor.load_and_preprocess()
    splits = preprocessor.create_splits(
        data_dict,
        test_size=config.get('test_size', 0.15),
        val_size=config.get('val_size', 0.15),
        random_state=config.get('seed', 42)
    )

    num_classes = len(preprocessor.label_mapping)
    iv5_input_dim = data_dict['iv5']['features'].shape[1]
    phmrc_input_dim = data_dict['phmrc']['features'].shape[1]

    # Create model
    model = DANN(
        iv5_dim=iv5_input_dim,
        phmrc_dim=phmrc_input_dim,
        num_classes=num_classes,
        encoder_hidden=encoder_hidden,
        encoder_output=encoder_output,
        embedding_dim=embedding_dim,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Trained for {checkpoint['epoch']} epochs")
    print(f"  Best metric: {checkpoint.get('best_metric', 'N/A'):.4f}")

    return model, preprocessor, splits, config, num_classes


def evaluate_classification(model, dataloader, device, dataset_id):
    """Evaluate classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            data = batch['features'].float().to(device)
            labels = batch['label'].to(device)
            dataset_ids = batch['dataset_id'].to(device)

            # Forward pass
            class_output, _, _ = model(data, dataset_ids, alpha=0.0)

            # Predictions
            _, predicted = torch.max(class_output, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def evaluate_domain_accuracy(model, dataloader, device):
    """
    Evaluate domain classification accuracy.

    TARGET: ~50% means domain-invariant features (good!)
    >80% means model can easily distinguish domains (bad - overfitting)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            data = batch['features'].float().to(device)
            dataset_ids = batch['dataset_id'].to(device)

            # Forward pass - use alpha=1.0 for domain evaluation
            _, domain_output, _ = model(data, dataset_ids, alpha=1.0)

            # Predictions
            _, predicted = torch.max(domain_output, 1)

            correct += (predicted == dataset_ids).sum().item()
            total += dataset_ids.size(0)

    accuracy = 100 * correct / total
    return accuracy


def confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    """Print confusion matrix in readable format."""
    print("\nConfusion Matrix:")
    print("-" * 80)

    # Header
    header = "True\\Pred".ljust(20)
    for name in class_names:
        header += name[:8].ljust(10)
    print(header)
    print("-" * 80)

    # Rows
    for i, name in enumerate(class_names):
        row = name[:18].ljust(20)
        for j in range(len(class_names)):
            row += str(cm[i, j]).ljust(10)
        print(row)
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description='Simple model evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device(prefer_cuda=True)

    # Load model
    model, preprocessor, splits, config, num_classes = load_model(args.checkpoint, device)

    # Get test dataloaders
    print("\nPreparing test data...")
    dataloaders = preprocessor.create_dataloaders(
        splits,
        batch_size=32,
        use_balanced_sampling=False  # No balancing for evaluation
    )

    iv5_test = dataloaders['iv5']['test']
    phmrc_test = dataloaders['phmrc']['test']

    # Combine for overall metrics
    from torch.utils.data import DataLoader, ConcatDataset
    combined_test = DataLoader(
        ConcatDataset([iv5_test.dataset, phmrc_test.dataset]),
        batch_size=32,
        shuffle=False
    )

    print("✓ Data prepared")

    # Run evaluation
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)

    # 1. Classification accuracy per dataset
    print("\n📊 Classification Accuracy:")
    print("-" * 80)

    iv5_acc, iv5_preds, iv5_labels = evaluate_classification(
        model, iv5_test, device, dataset_id=0
    )
    print(f"IV5 Test:    {iv5_acc:.2f}%")

    phmrc_acc, phmrc_preds, phmrc_labels = evaluate_classification(
        model, phmrc_test, device, dataset_id=1
    )
    print(f"PHMRC Test:  {phmrc_acc:.2f}%")

    avg_acc = (iv5_acc + phmrc_acc) / 2
    print(f"Average:     {avg_acc:.2f}%")

    # 2. Domain accuracy (CRITICAL METRIC)
    print("\n🎯 Domain Classification Accuracy (CRITICAL):")
    print("-" * 80)
    domain_acc = evaluate_domain_accuracy(model, combined_test, device)
    print(f"Domain Accuracy: {domain_acc:.2f}%")

    if 48 <= domain_acc <= 52:
        status = "✓ EXCELLENT - Domain invariance achieved!"
        color = "green"
    elif 40 <= domain_acc <= 60:
        status = "⚠ ACCEPTABLE - Reasonable domain invariance"
        color = "yellow"
    else:
        status = "✗ POOR - Model is overfitting to domain-specific features!"
        color = "red"

    print(f"Status: {status}")
    print(f"\nInterpretation:")
    print(f"  - Target range: 48-52% (random guessing = domain invariance)")
    print(f"  - Your result: {domain_acc:.2f}%")
    if domain_acc > 80:
        print(f"  ⚠ WARNING: Model can easily distinguish domains - NOT domain-invariant!")
    elif domain_acc < 40:
        print(f"  ⚠ WARNING: Suspiciously low - check implementation")
    else:
        print(f"  ✓ Model successfully learned domain-invariant features")

    # 3. Confusion matrices
    class_names = [k for k, v in sorted(preprocessor.label_mapping.items(), key=lambda x: x[1])]

    print("\n📋 IV5 Test Set Confusion Matrix:")
    iv5_cm = confusion_matrix(iv5_labels, iv5_preds, num_classes)
    print_confusion_matrix(iv5_cm, class_names)

    print("\n📋 PHMRC Test Set Confusion Matrix:")
    phmrc_cm = confusion_matrix(phmrc_labels, phmrc_preds, num_classes)
    print_confusion_matrix(phmrc_cm, class_names)

    # 4. Save results
    results = {
        'checkpoint': args.checkpoint,
        'test_results': {
            'iv5_accuracy': float(iv5_acc),
            'phmrc_accuracy': float(phmrc_acc),
            'average_accuracy': float(avg_acc),
            'domain_accuracy': float(domain_acc),
            'domain_invariance_status': status
        },
        'confusion_matrices': {
            'iv5': iv5_cm.tolist(),
            'phmrc': phmrc_cm.tolist()
        },
        'class_names': class_names
    }

    output_path = 'results/test_evaluation.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()

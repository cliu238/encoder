"""
Evaluation script for trained DANN models.

This script loads a trained model and performs comprehensive evaluation:
1. Embedding quality analysis
2. Cross-dataset transfer performance
3. Visualization generation
4. Detailed per-class metrics

Usage:
    python evaluate.py --checkpoint results/checkpoints/checkpoint_best.pt
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import VADataPreprocessor
from src.models.dann import DANN
from src.validation import EmbeddingValidator, TransferValidator
from src.utils import get_device, set_seed


class ModelEvaluator:
    """Comprehensive model evaluation tool."""

    def __init__(self, checkpoint_path, config=None):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config: Optional configuration dict (defaults used if None)
        """
        self.checkpoint_path = checkpoint_path
        self.config = config or self._default_config()

        set_seed(self.config['seed'])
        self.device = get_device(prefer_cuda=self.config.get('use_cuda', True))

        self.class_names = [
            'diarrhea', 'hiv', 'injury', 'malaria',
            'other_ch', 'other_infections', 'pneumonia'
        ]

        # Initialize components
        self.model = None
        self.preprocessor = None
        self.dataloaders = None

    def _default_config(self):
        """Default configuration."""
        return {
            'iv5_path': 'data/IV5_child_8categories.csv',
            'phmrc_path': 'data/PHMRC_child_8categories.csv',
            'test_size': 0.15,
            'val_size': 0.15,
            'batch_size': 32,
            'seed': 42,
            'use_cuda': True,
            'output_dir': 'results/evaluation'
        }

    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

        # Load data to get feature dimensions
        self.preprocessor = VADataPreprocessor(
            iv5_path=self.config['iv5_path'],
            phmrc_path=self.config['phmrc_path']
        )

        feature_dims = self.preprocessor.get_feature_dims()

        # Infer architecture from checkpoint weights
        state_dict = checkpoint['model_state_dict']

        # Infer encoder_hidden from first layer
        encoder_hidden = state_dict['iv5_encoder.encoder.0.weight'].shape[0]

        # Infer encoder_output from encoder's final layer
        encoder_output = state_dict['iv5_encoder.encoder.8.weight'].shape[0]

        # Infer embedding_dim from shared embedding layer
        embedding_dim = state_dict['shared_embedding.embedding.0.weight'].shape[0]

        print(f"  Inferred architecture: encoder_hidden={encoder_hidden}, encoder_output={encoder_output}, embedding_dim={embedding_dim}")

        # Create model with inferred architecture
        self.model = DANN(
            iv5_dim=feature_dims['iv5'],
            phmrc_dim=feature_dims['phmrc'],
            num_classes=len(self.class_names),
            encoder_hidden=encoder_hidden,
            encoder_output=encoder_output,
            embedding_dim=embedding_dim
        ).to(self.device)

        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.eval()

        print(f"✓ Model loaded successfully")
        print(f"  Trained for {checkpoint['epoch']+1} epochs")
        print(f"  Best metric: {checkpoint['metric']:.4f}")

    def prepare_data(self):
        """Prepare evaluation datasets."""
        print("\nPreparing evaluation data...")

        # Load and preprocess
        data = self.preprocessor.load_and_preprocess()

        # Create splits
        splits = self.preprocessor.create_splits(
            data,
            test_size=self.config['test_size'],
            val_size=self.config['val_size'],
            random_state=self.config['seed']
        )

        # Create dataloaders
        self.dataloaders = self.preprocessor.create_dataloaders(
            splits,
            batch_size=self.config['batch_size'],
            use_balanced_sampling=False  # No sampling for evaluation
        )

        print("✓ Data prepared successfully")

    def evaluate_embedding_quality(self):
        """Evaluate embedding space quality."""
        print("\n" + "="*80)
        print("EMBEDDING QUALITY EVALUATION")
        print("="*80)

        validator = EmbeddingValidator(self.model, self.device, self.class_names)

        # Evaluate
        metrics = validator.evaluate_embedding_quality(
            self.dataloaders['iv5']['test'],
            self.dataloaders['phmrc']['test']
        )

        # Create output directory
        os.makedirs(self.config['output_dir'], exist_ok=True)

        # Visualize embeddings (t-SNE)
        print("\nGenerating t-SNE visualization...")
        fig_tsne = validator.visualize_embeddings(
            [metrics['iv5_data'], metrics['phmrc_data']],
            save_path=os.path.join(self.config['output_dir'], 'embeddings_tsne.png'),
            method='tsne'
        )
        plt.close(fig_tsne)

        # Visualize embeddings (UMAP if available)
        try:
            from umap import UMAP
            print("Generating UMAP visualization...")
            fig_umap = validator.visualize_embeddings(
                [metrics['iv5_data'], metrics['phmrc_data']],
                save_path=os.path.join(self.config['output_dir'], 'embeddings_umap.png'),
                method='umap'
            )
            plt.close(fig_umap)
        except ImportError:
            print("UMAP not available, skipping UMAP visualization")

        return metrics

    def evaluate_within_dataset(self):
        """Evaluate within-dataset performance (baseline)."""
        print("\n" + "="*80)
        print("WITHIN-DATASET EVALUATION (BASELINE)")
        print("="*80)

        results = {}

        for dataset_name in ['iv5', 'phmrc']:
            print(f"\n{dataset_name.upper()} Test Set:")

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in self.dataloaders[dataset_name]['test']:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].cpu().numpy()

                    preds = self.model.predict(features, dataset=dataset_name)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(labels)

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            # Print classification report
            print(classification_report(
                all_labels, all_preds,
                target_names=self.class_names,
                digits=4
            ))

            # Store results
            from sklearn.metrics import accuracy_score, f1_score
            results[dataset_name] = {
                'accuracy': accuracy_score(all_labels, all_preds),
                'macro_f1': f1_score(all_labels, all_preds, average='macro'),
                'weighted_f1': f1_score(all_labels, all_preds, average='weighted'),
                'predictions': all_preds,
                'labels': all_labels
            }

        return results

    def evaluate_cross_dataset(self):
        """Evaluate cross-dataset transfer performance."""
        print("\n" + "="*80)
        print("CROSS-DATASET TRANSFER EVALUATION")
        print("="*80)

        validator = TransferValidator(self.model, self.device, self.class_names)

        # IV5 → PHMRC
        print("\n" + "-"*80)
        print("Transfer: IV5 → PHMRC (Train on IV5, Test on PHMRC)")
        print("-"*80)
        iv5_to_phmrc = validator.evaluate_transfer(
            self.dataloaders['iv5']['test'],
            self.dataloaders['phmrc']['test'],
            'IV5', 'PHMRC'
        )

        # PHMRC → IV5
        print("\n" + "-"*80)
        print("Transfer: PHMRC → IV5 (Train on PHMRC, Test on IV5)")
        print("-"*80)
        phmrc_to_iv5 = validator.evaluate_transfer(
            self.dataloaders['phmrc']['test'],
            self.dataloaders['iv5']['test'],
            'PHMRC', 'IV5'
        )

        # Save confusion matrices
        os.makedirs(self.config['output_dir'], exist_ok=True)

        fig1 = validator.plot_confusion_matrix(
            iv5_to_phmrc['confusion_matrix'],
            save_path=os.path.join(self.config['output_dir'], 'cm_iv5_to_phmrc.png')
        )
        plt.close(fig1)

        fig2 = validator.plot_confusion_matrix(
            phmrc_to_iv5['confusion_matrix'],
            save_path=os.path.join(self.config['output_dir'], 'cm_phmrc_to_iv5.png')
        )
        plt.close(fig2)

        return {
            'iv5_to_phmrc': iv5_to_phmrc,
            'phmrc_to_iv5': phmrc_to_iv5
        }

    def generate_summary_report(self, embedding_metrics, within_metrics, cross_metrics):
        """Generate comprehensive summary report."""
        print("\n" + "="*80)
        print("EVALUATION SUMMARY REPORT")
        print("="*80)

        # Create report
        report = []
        report.append("="*80)
        report.append("SHARED EMBEDDING SPACE EVALUATION REPORT")
        report.append("="*80)
        report.append("")

        # Model info
        report.append("MODEL INFORMATION")
        report.append("-"*80)
        report.append(f"Checkpoint: {self.checkpoint_path}")
        report.append(f"Device: {self.device}")
        report.append("")

        # Embedding quality
        report.append("EMBEDDING QUALITY METRICS")
        report.append("-"*80)
        report.append(f"Silhouette Score: {embedding_metrics['silhouette_score']:.4f}")
        report.append(f"  → Measures cluster separation (higher = better, range: -1 to 1)")
        report.append(f"5-NN Purity: {embedding_metrics['nn_purity']:.4f}")
        report.append(f"  → Fraction of neighbors with same label (higher = better)")
        report.append(f"Cross-Dataset 5-NN Agreement: {embedding_metrics['cross_dataset_nn_agreement']:.4f}")
        report.append(f"  → Label agreement for cross-dataset neighbors (higher = better)")
        report.append(f"Domain Classification Accuracy: {embedding_metrics['domain_classification_acc']:.4f}")
        report.append(f"  → Ability to distinguish datasets (closer to 0.50 = better)")
        report.append("")

        # Within-dataset performance
        report.append("WITHIN-DATASET PERFORMANCE (BASELINE)")
        report.append("-"*80)
        for dataset in ['iv5', 'phmrc']:
            report.append(f"{dataset.upper()}:")
            report.append(f"  Accuracy: {within_metrics[dataset]['accuracy']:.4f}")
            report.append(f"  Macro F1: {within_metrics[dataset]['macro_f1']:.4f}")
            report.append(f"  Weighted F1: {within_metrics[dataset]['weighted_f1']:.4f}")
        report.append("")

        # Cross-dataset transfer
        report.append("CROSS-DATASET TRANSFER PERFORMANCE")
        report.append("-"*80)
        report.append("IV5 → PHMRC (Train on IV5, Test on PHMRC):")
        report.append(f"  Accuracy: {cross_metrics['iv5_to_phmrc']['accuracy']:.4f}")
        report.append(f"  Macro F1: {cross_metrics['iv5_to_phmrc']['macro_f1']:.4f}")
        report.append(f"  Weighted F1: {cross_metrics['iv5_to_phmrc']['weighted_f1']:.4f}")
        report.append("")
        report.append("PHMRC → IV5 (Train on PHMRC, Test on IV5):")
        report.append(f"  Accuracy: {cross_metrics['phmrc_to_iv5']['accuracy']:.4f}")
        report.append(f"  Macro F1: {cross_metrics['phmrc_to_iv5']['macro_f1']:.4f}")
        report.append(f"  Weighted F1: {cross_metrics['phmrc_to_iv5']['weighted_f1']:.4f}")
        report.append("")

        # Interpretation
        report.append("INTERPRETATION GUIDE")
        report.append("-"*80)
        report.append("✓ Good shared embedding space:")
        report.append("  - High silhouette score (> 0.3)")
        report.append("  - High cross-dataset NN agreement (> 0.6)")
        report.append("  - Domain accuracy near 0.50 (domain confusion)")
        report.append("  - Strong cross-dataset transfer performance")
        report.append("")
        report.append("✗ Poor shared embedding space:")
        report.append("  - Low cross-dataset NN agreement (< 0.4)")
        report.append("  - High domain accuracy (> 0.8) = datasets still separable")
        report.append("  - Large gap between within-dataset and cross-dataset performance")
        report.append("")

        report.append("="*80)

        # Print report
        report_text = "\n".join(report)
        print(report_text)

        # Save report
        report_path = os.path.join(self.config['output_dir'], 'evaluation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\n✓ Report saved to: {report_path}")

        return report_text

    def run_full_evaluation(self):
        """Run complete evaluation pipeline."""
        # Load model
        self.load_model()

        # Prepare data
        self.prepare_data()

        # Run evaluations
        embedding_metrics = self.evaluate_embedding_quality()
        within_metrics = self.evaluate_within_dataset()
        cross_metrics = self.evaluate_cross_dataset()

        # Generate report
        report = self.generate_summary_report(
            embedding_metrics, within_metrics, cross_metrics
        )

        print("\n" + "="*80)
        print("EVALUATION COMPLETE!")
        print("="*80)
        print(f"Results saved to: {self.config['output_dir']}")

        return {
            'embedding': embedding_metrics,
            'within_dataset': within_metrics,
            'cross_dataset': cross_metrics,
            'report': report
        }


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained DANN model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='results/checkpoints/checkpoint_best.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/evaluation',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--iv5-path',
        type=str,
        default='data/IV5_child_8categories.csv',
        help='Path to IV5 dataset'
    )
    parser.add_argument(
        '--phmrc-path',
        type=str,
        default='data/PHMRC_child_8categories.csv',
        help='Path to PHMRC dataset'
    )

    args = parser.parse_args()

    # Create config
    config = {
        'iv5_path': args.iv5_path,
        'phmrc_path': args.phmrc_path,
        'output_dir': args.output_dir,
        'test_size': 0.15,
        'val_size': 0.15,
        'batch_size': 32,
        'seed': 42,
        'use_cuda': True
    }

    # Create evaluator
    evaluator = ModelEvaluator(args.checkpoint, config)

    # Run evaluation
    results = evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()

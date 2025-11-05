"""
Validation framework for shared embedding space evaluation.

Metrics:
1. Embedding Quality:
   - Silhouette score (intra-class vs inter-class distances)
   - t-SNE/UMAP visualization
   - Within-dataset vs cross-dataset nearest neighbor analysis
   - Domain classification accuracy (should approach 50%)

2. Cross-Dataset Transfer:
   - Train on IV5, test on PHMRC
   - Train on PHMRC, test on IV5
   - Comparison to within-dataset baselines
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, silhouette_score, classification_report
)
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, but fall back to t-SNE if not available
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available, will use t-SNE only")


class EmbeddingValidator:
    """Validator for embedding space quality."""

    def __init__(self, model, device='cpu', class_names=None):
        """
        Args:
            model: Trained DANN model
            device: Device to run validation on
            class_names: List of class names for visualization
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(7)]

    def extract_embeddings(self, dataloader, dataset_name='iv5'):
        """
        Extract embeddings for all samples in a dataloader.

        Args:
            dataloader: PyTorch dataloader
            dataset_name: Dataset name ('iv5' or 'phmrc')

        Returns:
            Dictionary with embeddings, labels, and dataset IDs
        """
        self.model.eval()
        embeddings_list = []
        labels_list = []
        dataset_ids_list = []

        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                labels = batch['label'].cpu().numpy()
                dataset_ids = batch['dataset_id'].cpu().numpy()

                # Get embeddings
                embeddings = self.model.encode(features, dataset=dataset_name)
                embeddings_list.append(embeddings.cpu().numpy())
                labels_list.append(labels)
                dataset_ids_list.append(dataset_ids)

        return {
            'embeddings': np.vstack(embeddings_list),
            'labels': np.concatenate(labels_list),
            'dataset_ids': np.concatenate(dataset_ids_list)
        }

    def compute_silhouette_score(self, embeddings, labels):
        """
        Compute silhouette score for embedding quality.

        Higher is better (range: -1 to 1)
        - Close to 1: Well-separated clusters
        - Close to 0: Overlapping clusters
        - Negative: Misassigned samples

        Args:
            embeddings: Embedding vectors
            labels: Class labels

        Returns:
            Silhouette score
        """
        if len(np.unique(labels)) < 2:
            return 0.0

        score = silhouette_score(embeddings, labels, metric='euclidean')
        return score

    def compute_nearest_neighbor_purity(self, embeddings, labels, k=5):
        """
        Compute k-nearest neighbor purity.

        Measures how many of the k nearest neighbors share the same class.

        Args:
            embeddings: Embedding vectors
            labels: Class labels
            k: Number of neighbors to consider

        Returns:
            Average purity score
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nbrs.fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # Exclude self (first neighbor)
        neighbor_indices = indices[:, 1:]

        # Compute purity
        purity_scores = []
        for i, neighbors in enumerate(neighbor_indices):
            neighbor_labels = labels[neighbors]
            purity = (neighbor_labels == labels[i]).mean()
            purity_scores.append(purity)

        return np.mean(purity_scores)

    def compute_cross_dataset_nn_ratio(self, embeddings_iv5, embeddings_phmrc,
                                       labels_iv5, labels_phmrc, k=5):
        """
        Compute cross-dataset nearest neighbor ratio.

        For each IV5 sample, find k nearest neighbors in PHMRC and check label agreement.
        High ratio indicates good domain alignment.

        Args:
            embeddings_iv5: IV5 embeddings
            embeddings_phmrc: PHMRC embeddings
            labels_iv5: IV5 labels
            labels_phmrc: PHMRC labels
            k: Number of neighbors

        Returns:
            Cross-dataset agreement ratio
        """
        # Find nearest PHMRC neighbors for each IV5 sample
        nbrs = NearestNeighbors(n_neighbors=k, metric='euclidean')
        nbrs.fit(embeddings_phmrc)
        distances, indices = nbrs.kneighbors(embeddings_iv5)

        # Check label agreement
        agreements = []
        for i, neighbors in enumerate(indices):
            neighbor_labels = labels_phmrc[neighbors]
            agreement = (neighbor_labels == labels_iv5[i]).mean()
            agreements.append(agreement)

        return np.mean(agreements)

    def compute_domain_confusion(self, embeddings, dataset_ids):
        """
        Train a simple classifier to distinguish datasets.

        Domain confusion is good - we want it to be hard to tell datasets apart.
        Accuracy close to 50% (random guessing) is ideal.

        Args:
            embeddings: Embedding vectors
            dataset_ids: Dataset labels (0=IV5, 1=PHMRC)

        Returns:
            Domain classification accuracy
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        # Use cross-validation to avoid overfitting
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, embeddings, dataset_ids, cv=5)

        return scores.mean()

    def visualize_embeddings(self, embeddings_dict_list, save_path=None, method='tsne'):
        """
        Visualize embeddings using t-SNE or UMAP.

        Args:
            embeddings_dict_list: List of dictionaries with 'embeddings', 'labels', 'dataset_ids'
            save_path: Path to save the plot
            method: 'tsne' or 'umap'
        """
        # Combine all embeddings
        all_embeddings = np.vstack([d['embeddings'] for d in embeddings_dict_list])
        all_labels = np.concatenate([d['labels'] for d in embeddings_dict_list])
        all_dataset_ids = np.concatenate([d['dataset_ids'] for d in embeddings_dict_list])

        # Reduce dimensionality
        print(f"Computing {method.upper()} projection...")
        if method == 'umap' and UMAP_AVAILABLE:
            reducer = UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
        else:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)

        embeddings_2d = reducer.fit_transform(all_embeddings)

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Color by class
        for class_idx in range(len(self.class_names)):
            mask = all_labels == class_idx
            axes[0].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=self.class_names[class_idx],
                alpha=0.6,
                s=20
            )
        axes[0].set_title(f'{method.upper()} Projection - Colored by Class', fontsize=14)
        axes[0].legend(loc='best')
        axes[0].set_xlabel(f'{method.upper()}-1')
        axes[0].set_ylabel(f'{method.upper()}-2')

        # Plot 2: Color by dataset
        dataset_names = ['IV5', 'PHMRC']
        colors = ['blue', 'red']
        for dataset_idx, (name, color) in enumerate(zip(dataset_names, colors)):
            mask = all_dataset_ids == dataset_idx
            axes[1].scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                label=name,
                alpha=0.6,
                s=20,
                color=color
            )
        axes[1].set_title(f'{method.upper()} Projection - Colored by Dataset', fontsize=14)
        axes[1].legend(loc='best')
        axes[1].set_xlabel(f'{method.upper()}-1')
        axes[1].set_ylabel(f'{method.upper()}-2')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")

        return fig

    def evaluate_embedding_quality(self, iv5_dataloader, phmrc_dataloader):
        """
        Comprehensive embedding quality evaluation.

        Args:
            iv5_dataloader: IV5 dataloader
            phmrc_dataloader: PHMRC dataloader

        Returns:
            Dictionary of metrics
        """
        print("Extracting embeddings...")
        iv5_data = self.extract_embeddings(iv5_dataloader, 'iv5')
        phmrc_data = self.extract_embeddings(phmrc_dataloader, 'phmrc')

        # Combine data
        all_embeddings = np.vstack([iv5_data['embeddings'], phmrc_data['embeddings']])
        all_labels = np.concatenate([iv5_data['labels'], phmrc_data['labels']])
        all_dataset_ids = np.concatenate([iv5_data['dataset_ids'], phmrc_data['dataset_ids']])

        print("\nComputing embedding quality metrics...")

        # Silhouette score
        silhouette = self.compute_silhouette_score(all_embeddings, all_labels)
        print(f"  Silhouette score: {silhouette:.4f}")

        # Nearest neighbor purity
        nn_purity = self.compute_nearest_neighbor_purity(all_embeddings, all_labels, k=5)
        print(f"  5-NN purity: {nn_purity:.4f}")

        # Cross-dataset NN agreement
        cross_nn_ratio = self.compute_cross_dataset_nn_ratio(
            iv5_data['embeddings'], phmrc_data['embeddings'],
            iv5_data['labels'], phmrc_data['labels'], k=5
        )
        print(f"  Cross-dataset 5-NN agreement: {cross_nn_ratio:.4f}")

        # Domain confusion
        domain_acc = self.compute_domain_confusion(all_embeddings, all_dataset_ids)
        print(f"  Domain classification accuracy: {domain_acc:.4f} (ideal: 0.50)")

        return {
            'silhouette_score': silhouette,
            'nn_purity': nn_purity,
            'cross_dataset_nn_agreement': cross_nn_ratio,
            'domain_classification_acc': domain_acc,
            'iv5_data': iv5_data,
            'phmrc_data': phmrc_data
        }


class TransferValidator:
    """Validator for cross-dataset transfer performance."""

    def __init__(self, model, device='cpu', class_names=None):
        """
        Args:
            model: Trained DANN model
            device: Device to run validation on
            class_names: List of class names
        """
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names or [f"Class_{i}" for i in range(7)]

    def evaluate_transfer(self, source_loader, target_loader,
                         source_name='IV5', target_name='PHMRC'):
        """
        Evaluate transfer performance from source to target dataset.

        Args:
            source_loader: Dataloader for source dataset
            target_loader: Dataloader for target dataset
            source_name: Source dataset name
            target_name: Target dataset name

        Returns:
            Dictionary of transfer metrics
        """
        self.model.eval()

        print(f"\nEvaluating transfer: {source_name} → {target_name}")

        # Collect predictions and labels
        all_preds = []
        all_labels = []

        dataset_name = target_name.lower()

        with torch.no_grad():
            for batch in target_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].cpu().numpy()

                # Make predictions
                preds = self.model.predict(features, dataset=dataset_name)
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels)

        # Concatenate results
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Weighted F1: {weighted_f1:.4f}")

        # Per-class metrics
        print(f"\n  Per-class F1 scores:")
        per_class_f1 = f1_score(all_labels, all_preds, average=None)
        for i, (class_name, f1) in enumerate(zip(self.class_names, per_class_f1)):
            print(f"    {class_name}: {f1:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_f1': per_class_f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }

    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")

        return plt.gcf()


def test_validation():
    """Test validation framework."""
    print("Testing validation framework...")

    # Create dummy model and data
    from src.models.dann import DANN

    model = DANN(iv5_dim=353, phmrc_dim=109, num_classes=7)

    # Create dummy dataloaders
    from torch.utils.data import TensorDataset, DataLoader

    # IV5 dummy data
    iv5_features = torch.randn(100, 353)
    iv5_labels = torch.randint(0, 7, (100,))
    iv5_dataset_ids = torch.zeros(100, dtype=torch.long)
    iv5_dataset = TensorDataset(iv5_features, iv5_labels, iv5_dataset_ids)
    iv5_loader = DataLoader(iv5_dataset, batch_size=16)

    # PHMRC dummy data
    phmrc_features = torch.randn(100, 109)
    phmrc_labels = torch.randint(0, 7, (100,))
    phmrc_dataset_ids = torch.ones(100, dtype=torch.long)
    phmrc_dataset = TensorDataset(phmrc_features, phmrc_labels, phmrc_dataset_ids)
    phmrc_loader = DataLoader(phmrc_dataset, batch_size=16)

    # Modify loaders to match expected format
    class DummyLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for features, labels, dataset_ids in self.loader:
                yield {
                    'features': features,
                    'label': labels,
                    'dataset_id': dataset_ids
                }

    iv5_loader = DummyLoader(iv5_loader)
    phmrc_loader = DummyLoader(phmrc_loader)

    # Test embedding validator
    print("\nTesting EmbeddingValidator...")
    embedding_validator = EmbeddingValidator(model, class_names=[
        'diarrhea', 'hiv', 'injury', 'malaria', 'other_ch', 'other_infections', 'pneumonia'
    ])

    metrics = embedding_validator.evaluate_embedding_quality(iv5_loader, phmrc_loader)
    print(f"Embedding quality metrics: {metrics.keys()}")

    # Test transfer validator
    print("\nTesting TransferValidator...")
    transfer_validator = TransferValidator(model, class_names=[
        'diarrhea', 'hiv', 'injury', 'malaria', 'other_ch', 'other_infections', 'pneumonia'
    ])

    transfer_metrics = transfer_validator.evaluate_transfer(
        iv5_loader, phmrc_loader, 'IV5', 'PHMRC'
    )
    print(f"Transfer metrics: {transfer_metrics.keys()}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_validation()

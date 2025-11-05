"""
Data preprocessing module for cross-dataset verbal autopsy classification.

Handles:
- Loading IV5 and PHMRC datasets
- Label harmonization (merging severe_malnutrition into other_ch)
- Missing value encoding (3-state: Y=1, N=0, Missing=-1)
- Stratified train/val/test splits
- PyTorch datasets with balanced sampling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class VADataset(Dataset):
    """PyTorch Dataset for verbal autopsy data."""

    def __init__(self, features, labels, dataset_id):
        """
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
            dataset_id: int (0 for IV5, 1 for PHMRC)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.dataset_id = torch.LongTensor([dataset_id] * len(labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'dataset_id': self.dataset_id[idx]
        }


class VADataPreprocessor:
    """Preprocessor for verbal autopsy datasets."""

    def __init__(self, iv5_path, phmrc_path):
        """
        Args:
            iv5_path: Path to IV5 CSV file
            phmrc_path: Path to PHMRC CSV file
        """
        self.iv5_path = iv5_path
        self.phmrc_path = phmrc_path
        self.label_mapping = {}
        self.class_names = []

    def load_and_preprocess(self):
        """
        Load both datasets and preprocess them.

        Returns:
            dict with keys 'iv5' and 'phmrc', each containing:
                - 'features': numpy array of features
                - 'labels': numpy array of encoded labels
                - 'label_names': list of original label names
        """
        print("Loading datasets...")

        # Load IV5
        iv5_df = pd.read_csv(self.iv5_path)
        print(f"  IV5: {len(iv5_df)} samples, {len(iv5_df.columns)} columns")

        # Load PHMRC
        phmrc_df = pd.read_csv(self.phmrc_path)
        print(f"  PHMRC: {len(phmrc_df)} samples, {len(phmrc_df.columns)} columns")

        # Merge severe_malnutrition into other_ch for IV5
        print("\nMerging 'severe_malnutrition' into 'other_ch' for IV5...")
        iv5_df['broader_category'] = iv5_df['broader_category'].replace(
            'severe_malnutrition', 'other_ch'
        )
        severe_mal_count = (iv5_df['broader_category'] == 'severe_malnutrition').sum()
        print(f"  Merged {291} severe_malnutrition samples into other_ch")

        # Get shared labels
        iv5_labels = set(iv5_df['broader_category'].unique())
        phmrc_labels = set(phmrc_df['broader_category'].unique())
        shared_labels = sorted(iv5_labels & phmrc_labels)

        print(f"\nShared labels ({len(shared_labels)}): {shared_labels}")

        # Create label encoding
        self.class_names = shared_labels
        self.label_mapping = {label: idx for idx, label in enumerate(shared_labels)}
        print(f"Label mapping: {self.label_mapping}")

        # Filter to shared labels only
        iv5_df = iv5_df[iv5_df['broader_category'].isin(shared_labels)]
        phmrc_df = phmrc_df[phmrc_df['broader_category'].isin(shared_labels)]

        print(f"\nAfter filtering to shared labels:")
        print(f"  IV5: {len(iv5_df)} samples")
        print(f"  PHMRC: {len(phmrc_df)} samples")

        # Process IV5
        iv5_features, iv5_labels, iv5_label_names = self._process_iv5(iv5_df)

        # Process PHMRC
        phmrc_features, phmrc_labels, phmrc_label_names = self._process_phmrc(phmrc_df)

        # Print class distribution
        self._print_class_distribution(iv5_labels, phmrc_labels)

        return {
            'iv5': {
                'features': iv5_features,
                'labels': iv5_labels,
                'label_names': iv5_label_names
            },
            'phmrc': {
                'features': phmrc_features,
                'labels': phmrc_labels,
                'label_names': phmrc_label_names
            }
        }

    def _process_iv5(self, df):
        """Process IV5 dataset."""
        # Separate features and labels
        labels = df['broader_category'].values
        features_df = df.drop(columns=['broader_category'])

        # Encode features: Y=1, N=0, .=-1
        features = features_df.replace({'Y': 1, 'N': 0, '.': -1}).values.astype(np.float32)

        # Encode labels
        encoded_labels = np.array([self.label_mapping[label] for label in labels])

        return features, encoded_labels, labels

    def _process_phmrc(self, df):
        """Process PHMRC dataset with 3-state encoding for missing values."""
        # Separate features and labels
        labels = df['broader_category'].values
        features_df = df.drop(columns=['broader_category'])

        # Encode features: Y=1, .=0, NaN/empty=-1 (missing)
        # First replace Y and .
        features_df = features_df.replace({'Y': 1, '.': 0})

        # Convert to numeric, NaN becomes NaN
        features = features_df.apply(pd.to_numeric, errors='coerce').values

        # Replace NaN with -1 (missing indicator)
        features = np.nan_to_num(features, nan=-1).astype(np.float32)

        # Encode labels
        encoded_labels = np.array([self.label_mapping[label] for label in labels])

        return features, encoded_labels, labels

    def _print_class_distribution(self, iv5_labels, phmrc_labels):
        """Print class distribution for both datasets."""
        print("\nClass distribution:")
        print(f"{'Class':<20} {'IV5':>8} {'PHMRC':>8} {'Total':>8}")
        print("-" * 50)

        for class_name in self.class_names:
            class_idx = self.label_mapping[class_name]
            iv5_count = (iv5_labels == class_idx).sum()
            phmrc_count = (phmrc_labels == class_idx).sum()
            total = iv5_count + phmrc_count
            print(f"{class_name:<20} {iv5_count:>8} {phmrc_count:>8} {total:>8}")

    def create_splits(self, data_dict, test_size=0.15, val_size=0.15, random_state=42):
        """
        Create stratified train/val/test splits for both datasets.

        Args:
            data_dict: Dictionary returned by load_and_preprocess()
            test_size: Fraction of data for test set
            val_size: Fraction of training data for validation set
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with train/val/test splits for both datasets
        """
        print(f"\nCreating splits (train/val/test)...")

        splits = {}

        for dataset_name in ['iv5', 'phmrc']:
            features = data_dict[dataset_name]['features']
            labels = data_dict[dataset_name]['labels']

            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                features, labels,
                test_size=test_size,
                stratify=labels,
                random_state=random_state
            )

            # Second split: train vs val
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size / (1 - test_size),  # Adjust for the remaining data
                stratify=y_temp,
                random_state=random_state
            )

            splits[dataset_name] = {
                'train': {'features': X_train, 'labels': y_train},
                'val': {'features': X_val, 'labels': y_val},
                'test': {'features': X_test, 'labels': y_test}
            }

            print(f"  {dataset_name.upper()}: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")

        return splits

    def create_dataloaders(self, splits, batch_size=32, use_balanced_sampling=True):
        """
        Create PyTorch DataLoaders with optional balanced sampling.

        Args:
            splits: Dictionary returned by create_splits()
            batch_size: Batch size for training
            use_balanced_sampling: Whether to use balanced sampling for training

        Returns:
            Dictionary of DataLoaders for each dataset and split
        """
        print(f"\nCreating DataLoaders (batch_size={batch_size}, balanced={use_balanced_sampling})...")

        dataloaders = {}

        for dataset_name, dataset_id in [('iv5', 0), ('phmrc', 1)]:
            dataloaders[dataset_name] = {}

            for split_name in ['train', 'val', 'test']:
                features = splits[dataset_name][split_name]['features']
                labels = splits[dataset_name][split_name]['labels']

                # Create dataset
                dataset = VADataset(features, labels, dataset_id)

                # Create sampler for balanced training
                sampler = None
                shuffle = (split_name == 'train')

                if split_name == 'train' and use_balanced_sampling:
                    # Compute class weights
                    class_weights = compute_class_weight(
                        'balanced',
                        classes=np.unique(labels),
                        y=labels
                    )

                    # Create sample weights
                    sample_weights = np.array([class_weights[label] for label in labels])
                    sample_weights = torch.FloatTensor(sample_weights)

                    # Create weighted sampler
                    sampler = WeightedRandomSampler(
                        weights=sample_weights,
                        num_samples=len(sample_weights),
                        replacement=True
                    )
                    shuffle = False  # Sampler handles shuffling

                # Create dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    num_workers=0  # Set to 0 for simplicity, increase for performance
                )

                dataloaders[dataset_name][split_name] = dataloader

        print("DataLoaders created successfully!")
        return dataloaders

    def get_feature_dims(self):
        """Get feature dimensions for IV5 and PHMRC."""
        iv5_df = pd.read_csv(self.iv5_path)
        phmrc_df = pd.read_csv(self.phmrc_path)

        iv5_dim = len(iv5_df.columns) - 1  # Exclude broader_category
        phmrc_dim = len(phmrc_df.columns) - 1

        return {'iv5': iv5_dim, 'phmrc': phmrc_dim}


def main():
    """Example usage."""
    # Initialize preprocessor
    preprocessor = VADataPreprocessor(
        iv5_path='data/IV5_child_8categories.csv',
        phmrc_path='data/PHMRC_child_8categories.csv'
    )

    # Load and preprocess data
    data = preprocessor.load_and_preprocess()

    # Create splits
    splits = preprocessor.create_splits(data)

    # Create dataloaders
    dataloaders = preprocessor.create_dataloaders(splits, batch_size=32)

    # Get feature dimensions
    feature_dims = preprocessor.get_feature_dims()
    print(f"\nFeature dimensions: {feature_dims}")

    # Test dataloader
    print("\nTesting dataloader...")
    batch = next(iter(dataloaders['iv5']['train']))
    print(f"Batch keys: {batch.keys()}")
    print(f"Features shape: {batch['features'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    print(f"Dataset IDs shape: {batch['dataset_id'].shape}")

    print("\nPreprocessing completed successfully!")


if __name__ == '__main__':
    main()

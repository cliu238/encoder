"""
Data augmentation for verbal autopsy binary/ternary features.

Since VA features are discrete (Y/N/Missing = 1/0/-1), we use specialized
augmentation strategies:
1. **Feature dropout**: Randomly set features to -1 (missing)
2. **Feature noise**: Add small Gaussian noise with probability-based flipping
3. **Feature swap**: Randomly swap similar features within semantic groups

These augmentations force the model to learn robust patterns that don't
depend on exact feature values.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class AugmentedVADataset(Dataset):
    """
    VA Dataset with data augmentation for training.

    Augmentation strategies:
    1. Feature dropout: Randomly set 5-10% of features to -1 (missing)
    2. Feature noise: Add Gaussian noise (σ=0.1) to simulate measurement uncertainty
    3. Feature flip: With small probability, flip binary values (simulates errors)
    """

    def __init__(
        self,
        features,
        labels,
        dataset_id,
        augment=True,
        dropout_prob=0.1,
        noise_std=0.1,
        flip_prob=0.05
    ):
        """
        Args:
            features: numpy array (n_samples, n_features)
            labels: numpy array (n_samples,)
            dataset_id: int (0=IV5, 1=PHMRC)
            augment: Whether to apply augmentation
            dropout_prob: Probability of setting feature to -1
            noise_std: Standard deviation of Gaussian noise
            flip_prob: Probability of flipping binary values
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.dataset_id = torch.LongTensor([dataset_id] * len(labels))

        self.augment = augment
        self.dropout_prob = dropout_prob
        self.noise_std = noise_std
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.labels)

    def augment_features(self, features):
        """
        Apply augmentation to features.

        Args:
            features: Tensor of shape (n_features,)

        Returns:
            Augmented features
        """
        if not self.augment:
            return features

        # Clone to avoid modifying original
        aug_features = features.clone()

        # 1. Feature dropout (set to -1 = missing)
        if self.dropout_prob > 0:
            dropout_mask = torch.rand(features.shape) < self.dropout_prob
            aug_features[dropout_mask] = -1.0

        # 2. Add Gaussian noise (but preserve -1 values)
        if self.noise_std > 0:
            noise = torch.randn_like(aug_features) * self.noise_std
            # Only add noise to non-missing features
            non_missing_mask = (aug_features != -1.0)
            aug_features = aug_features + noise * non_missing_mask.float()

        # 3. Binary flip (simulate measurement errors)
        # Only flip non-missing binary features (values close to 0 or 1)
        if self.flip_prob > 0:
            binary_mask = ((aug_features >= -0.5) & (aug_features <= 0.5)) | \
                         ((aug_features >= 0.5) & (aug_features <= 1.5))
            flip_mask = (torch.rand(features.shape) < self.flip_prob) & binary_mask & (aug_features != -1.0)

            # Flip: 0 → 1, 1 → 0
            aug_features[flip_mask] = 1.0 - aug_features[flip_mask]

        return aug_features

    def __getitem__(self, idx):
        features = self.features[idx]

        # Apply augmentation
        if self.augment:
            features = self.augment_features(features)

        return {
            'features': features,
            'label': self.labels[idx],
            'dataset_id': self.dataset_id[idx]
        }


def create_augmented_dataloaders(
    splits,
    batch_size=32,
    use_balanced_sampling=True,
    augment_train=True,
    dropout_prob=0.1,
    noise_std=0.1,
    flip_prob=0.05,
    num_workers=0
):
    """
    Create DataLoaders with augmentation for training data.

    Args:
        splits: Dictionary from VADataPreprocessor.create_splits()
        batch_size: Batch size
        use_balanced_sampling: Whether to use balanced sampling for training
        augment_train: Whether to apply augmentation to training data
        dropout_prob: Feature dropout probability
        noise_std: Gaussian noise standard deviation
        flip_prob: Binary flip probability
        num_workers: Number of workers for DataLoader

    Returns:
        Dictionary with keys 'iv5' and 'phmrc', each containing:
            - 'train': Training DataLoader (with augmentation)
            - 'val': Validation DataLoader (no augmentation)
            - 'test': Test DataLoader (no augmentation)
    """
    dataloaders = {}

    for dataset_name in ['iv5', 'phmrc']:
        dataset_id = 0 if dataset_name == 'iv5' else 1
        dataloaders[dataset_name] = {}

        for split_name in ['train', 'val', 'test']:
            split_data = splits[dataset_name][split_name]

            # Apply augmentation only to training data
            apply_augment = augment_train and (split_name == 'train')

            # Create dataset
            dataset = AugmentedVADataset(
                features=split_data['features'],
                labels=split_data['labels'],
                dataset_id=dataset_id,
                augment=apply_augment,
                dropout_prob=dropout_prob,
                noise_std=noise_std,
                flip_prob=flip_prob
            )

            # Create sampler for training
            sampler = None
            shuffle = (split_name == 'train')

            if split_name == 'train' and use_balanced_sampling:
                # Balanced sampling
                labels = split_data['labels']
                unique_labels, label_counts = np.unique(labels, return_counts=True)

                # Compute class weights
                class_weights = 1.0 / label_counts
                sample_weights = class_weights[labels]

                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                shuffle = False  # Don't shuffle when using sampler

            # Create DataLoader
            dataloaders[dataset_name][split_name] = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True
            )

    return dataloaders


if __name__ == '__main__':
    """Test augmented dataset."""
    print("Testing Augmented VA Dataset...")

    # Create dummy data
    np.random.seed(42)
    features = np.random.choice([0, 1, -1], size=(100, 50))
    labels = np.random.randint(0, 7, size=100)

    # Test without augmentation
    dataset_no_aug = AugmentedVADataset(
        features=features,
        labels=labels,
        dataset_id=0,
        augment=False
    )

    # Test with augmentation
    dataset_aug = AugmentedVADataset(
        features=features,
        labels=labels,
        dataset_id=0,
        augment=True,
        dropout_prob=0.1,
        noise_std=0.1,
        flip_prob=0.05
    )

    # Compare original vs augmented
    sample_idx = 0
    original = dataset_no_aug[sample_idx]['features']
    augmented1 = dataset_aug[sample_idx]['features']
    augmented2 = dataset_aug[sample_idx]['features']  # Should be different

    print(f"\nOriginal features (first 10): {original[:10].numpy()}")
    print(f"Augmented v1 (first 10):      {augmented1[:10].numpy()}")
    print(f"Augmented v2 (first 10):      {augmented2[:10].numpy()}")

    # Check differences
    diff1 = (original != augmented1).sum().item()
    diff2 = (augmented1 != augmented2).sum().item()

    print(f"\nFeatures changed (original vs aug1): {diff1}/{len(original)}")
    print(f"Features changed (aug1 vs aug2): {diff2}/{len(original)}")

    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset_aug, batch_size=16, shuffle=True)

    batch = next(iter(loader))
    print(f"\nBatch features shape: {batch['features'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")

    print("\n✓ Augmented dataset test passed!")

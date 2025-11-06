"""
Maximum Mean Discrepancy (MMD) based domain adaptation model.

MMD is an alternative to DANN that directly minimizes the distance between
source and target domain distributions in the embedding space. It's often
more stable than adversarial training.

Key differences from DANN:
- No adversarial training (no gradient reversal)
- Direct distribution matching via kernel methods
- More stable training dynamics
- Better suited for small datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianKernel(nn.Module):
    """Gaussian RBF kernel for MMD computation."""

    def __init__(self, sigmas=None):
        """
        Args:
            sigmas: List of kernel bandwidths. If None, uses [0.01, 0.1, 1, 10, 100]
        """
        super().__init__()
        if sigmas is None:
            sigmas = [0.01, 0.1, 1, 10, 100]
        self.sigmas = sigmas

    def forward(self, x, y):
        """
        Compute multi-kernel MMD between x and y.

        Args:
            x: Tensor of shape (n, d)
            y: Tensor of shape (m, d)

        Returns:
            MMD distance (scalar)
        """
        # Compute pairwise distances
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        # ||x - x'||^2
        dxx = rx.t() + rx - 2. * xx
        # ||y - y'||^2
        dyy = ry.t() + ry - 2. * yy
        # ||x - y||^2
        dxy = rx.t() + ry - 2. * xy

        # Multi-kernel MMD
        mmd = torch.zeros(1, device=x.device)

        for sigma in self.sigmas:
            # Gaussian kernels
            kxx = torch.exp(-dxx / (2 * sigma))
            kyy = torch.exp(-dyy / (2 * sigma))
            kxy = torch.exp(-dxy / (2 * sigma))

            # MMD = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
            mmd += kxx.mean() + kyy.mean() - 2 * kxy.mean()

        return mmd


class Encoder(nn.Module):
    """Dataset-specific encoder."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.network(x)


class MMDModel(nn.Module):
    """
    MMD-based domain adaptation model for VA cross-dataset classification.

    Architecture:
        IV5/PHMRC → Dataset-specific encoders → Shared embedding → Classifier

    Training objective:
        L = L_cls + lambda_mmd * MMD(embedding_iv5, embedding_phmrc)

    where:
        - L_cls: Classification loss (CrossEntropy)
        - MMD: Maximum Mean Discrepancy between domain embeddings
    """

    def __init__(
        self,
        iv5_dim,
        phmrc_dim,
        num_classes,
        encoder_hidden=256,
        encoder_output=128,
        embedding_dim=64,
        dropout=0.3,
        kernel_sigmas=None
    ):
        """
        Args:
            iv5_dim: IV5 input dimension (353)
            phmrc_dim: PHMRC input dimension (109)
            num_classes: Number of classes (7)
            encoder_hidden: Hidden layer size for encoders
            encoder_output: Output dimension of encoders
            embedding_dim: Shared embedding dimension
            dropout: Dropout rate
            kernel_sigmas: Kernel bandwidths for MMD (list of floats)
        """
        super().__init__()

        # Dataset-specific encoders
        self.iv5_encoder = Encoder(iv5_dim, encoder_hidden, encoder_output, dropout)
        self.phmrc_encoder = Encoder(phmrc_dim, encoder_hidden, encoder_output, dropout)

        # Shared embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(encoder_output, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes)
        )

        # MMD kernel
        self.mmd_kernel = GaussianKernel(kernel_sigmas)

        # Store dimensions
        self.iv5_dim = iv5_dim
        self.phmrc_dim = phmrc_dim
        self.num_classes = num_classes

    def encode(self, features, dataset):
        """
        Encode features through dataset-specific encoder.

        Args:
            features: Input features
            dataset: 'iv5' or 'phmrc'

        Returns:
            Encoded features
        """
        if dataset == 'iv5':
            return self.iv5_encoder(features)
        elif dataset == 'phmrc':
            return self.phmrc_encoder(features)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def get_embedding(self, encoded):
        """Get shared embedding from encoded features."""
        return self.embedding(encoded)

    def forward(self, iv5_features=None, phmrc_features=None, compute_mmd=True):
        """
        Forward pass.

        Args:
            iv5_features: IV5 batch (optional)
            phmrc_features: PHMRC batch (optional)
            compute_mmd: Whether to compute MMD loss

        Returns:
            Dictionary with:
                - class_logits: Classification logits
                - embeddings: Shared embeddings
                - mmd_loss: MMD loss (if compute_mmd=True)
        """
        embeddings = []

        # Encode IV5
        if iv5_features is not None:
            iv5_encoded = self.iv5_encoder(iv5_features)
            iv5_embedding = self.embedding(iv5_encoded)
            embeddings.append(iv5_embedding)

        # Encode PHMRC
        if phmrc_features is not None:
            phmrc_encoded = self.phmrc_encoder(phmrc_features)
            phmrc_embedding = self.embedding(phmrc_encoded)
            embeddings.append(phmrc_embedding)

        # Concatenate embeddings
        all_embeddings = torch.cat(embeddings, dim=0)

        # Classify
        class_logits = self.classifier(all_embeddings)

        # Compute MMD loss
        mmd_loss = None
        if compute_mmd and len(embeddings) == 2:
            mmd_loss = self.mmd_kernel(embeddings[0], embeddings[1])

        return {
            'class_logits': class_logits,
            'embeddings': all_embeddings,
            'mmd_loss': mmd_loss
        }

    def predict(self, features, dataset):
        """
        Predict class labels.

        Args:
            features: Input features
            dataset: 'iv5' or 'phmrc'

        Returns:
            Predicted class labels
        """
        encoded = self.encode(features, dataset)
        embedding = self.get_embedding(encoded)
        logits = self.classifier(embedding)
        return torch.argmax(logits, dim=1)


if __name__ == '__main__':
    """Test MMD model."""
    print("Testing MMD Model...")

    # Create model
    model = MMDModel(
        iv5_dim=353,
        phmrc_dim=109,
        num_classes=7,
        encoder_hidden=128,
        encoder_output=64,
        embedding_dim=32,
        dropout=0.5
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 32
    iv5_features = torch.randn(batch_size, 353)
    phmrc_features = torch.randn(batch_size, 109)

    output = model(iv5_features, phmrc_features, compute_mmd=True)

    print(f"Class logits shape: {output['class_logits'].shape}")
    print(f"Embeddings shape: {output['embeddings'].shape}")
    print(f"MMD loss: {output['mmd_loss'].item():.4f}")

    # Test prediction
    preds = model.predict(iv5_features, 'iv5')
    print(f"Predictions shape: {preds.shape}")

    print("\n✓ MMD model test passed!")

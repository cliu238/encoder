"""
Domain-Adversarial Neural Network (DANN) for cross-dataset verbal autopsy classification.

Architecture:
- Dataset-specific encoders (IV5 and PHMRC)
- Shared embedding space (domain-invariant representations)
- Classification head for predicting broader_category
- Domain discriminator with gradient reversal for domain adaptation

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer.

    Passes input forward unchanged, but reverses gradients during backprop.
    This makes the encoder learn features that fool the domain discriminator.
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """Forward pass: identity function."""
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: reverse and scale gradients."""
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversalLayer(nn.Module):
    """Gradient reversal layer wrapper."""

    def __init__(self, lambda_=1.0):
        """
        Args:
            lambda_: Scaling factor for gradient reversal (default 1.0)
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class IV5Encoder(nn.Module):
    """Encoder for IV5 dataset (353 features)."""

    def __init__(self, input_dim=353, hidden_dim=256, output_dim=128, dropout=0.3):
        """
        Args:
            input_dim: Number of input features (353 for IV5)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
        """
        super(IV5Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class PHMRCEncoder(nn.Module):
    """Encoder for PHMRC dataset (109 features)."""

    def __init__(self, input_dim=109, hidden_dim=256, output_dim=128, dropout=0.3):
        """
        Args:
            input_dim: Number of input features (109 for PHMRC)
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
        """
        super(PHMRCEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class SharedEmbedding(nn.Module):
    """Shared embedding layer that creates domain-invariant representations."""

    def __init__(self, input_dim=128, output_dim=64, dropout=0.2):
        """
        Args:
            input_dim: Input dimension from encoders
            output_dim: Output embedding dimension (shared space)
            dropout: Dropout rate
        """
        super(SharedEmbedding, self).__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.embedding(x)


class ClassifierHead(nn.Module):
    """Classification head for predicting broader_category (7 classes)."""

    def __init__(self, input_dim=64, num_classes=7, dropout=0.2):
        """
        Args:
            input_dim: Input dimension from shared embedding
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super(ClassifierHead, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class DomainDiscriminator(nn.Module):
    """Domain discriminator for distinguishing IV5 vs PHMRC (with gradient reversal)."""

    def __init__(self, input_dim=64, dropout=0.2):
        """
        Args:
            input_dim: Input dimension from shared embedding
            dropout: Dropout rate
        """
        super(DomainDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(32, 2)  # Binary: IV5 (0) vs PHMRC (1)
        )

    def forward(self, x):
        return self.discriminator(x)


class DANN(nn.Module):
    """
    Complete Domain-Adversarial Neural Network.

    Combines dataset-specific encoders, shared embedding, classifier, and
    domain discriminator for cross-dataset learning.
    """

    def __init__(self, iv5_dim=353, phmrc_dim=109, num_classes=7,
                 encoder_hidden=256, encoder_output=128,
                 embedding_dim=64, dropout=0.3, lambda_domain=1.0):
        """
        Args:
            iv5_dim: IV5 feature dimension
            phmrc_dim: PHMRC feature dimension
            num_classes: Number of classes for classification
            encoder_hidden: Hidden dimension for encoders
            encoder_output: Output dimension for encoders
            embedding_dim: Shared embedding dimension
            dropout: Dropout rate
            lambda_domain: Gradient reversal scaling factor
        """
        super(DANN, self).__init__()

        # Dataset-specific encoders
        self.iv5_encoder = IV5Encoder(
            input_dim=iv5_dim,
            hidden_dim=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout
        )

        self.phmrc_encoder = PHMRCEncoder(
            input_dim=phmrc_dim,
            hidden_dim=encoder_hidden,
            output_dim=encoder_output,
            dropout=dropout
        )

        # Shared components
        self.shared_embedding = SharedEmbedding(
            input_dim=encoder_output,
            output_dim=embedding_dim,
            dropout=dropout
        )

        self.classifier = ClassifierHead(
            input_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout
        )

        self.gradient_reversal = GradientReversalLayer(lambda_=lambda_domain)

        self.domain_discriminator = DomainDiscriminator(
            input_dim=embedding_dim,
            dropout=dropout
        )

    def forward(self, iv5_features=None, phmrc_features=None, alpha=1.0):
        """
        Forward pass through the network.

        Args:
            iv5_features: IV5 input features (batch_size, 353)
            phmrc_features: PHMRC input features (batch_size, 109)
            alpha: Domain adaptation weight (increases during training)

        Returns:
            Dictionary containing:
                - 'embeddings': Shared embeddings
                - 'class_logits': Classification logits
                - 'domain_logits': Domain classification logits
                - 'dataset_id': Which dataset (0=IV5, 1=PHMRC)
        """
        embeddings_list = []
        dataset_ids = []

        # Process IV5 data
        if iv5_features is not None:
            iv5_encoded = self.iv5_encoder(iv5_features)
            iv5_embedding = self.shared_embedding(iv5_encoded)
            embeddings_list.append(iv5_embedding)
            dataset_ids.extend([0] * len(iv5_features))

        # Process PHMRC data
        if phmrc_features is not None:
            phmrc_encoded = self.phmrc_encoder(phmrc_features)
            phmrc_embedding = self.shared_embedding(phmrc_encoded)
            embeddings_list.append(phmrc_embedding)
            dataset_ids.extend([1] * len(phmrc_features))

        # Concatenate all embeddings
        embeddings = torch.cat(embeddings_list, dim=0)

        # Classification head
        class_logits = self.classifier(embeddings)

        # Domain discrimination (with gradient reversal)
        # Update gradient reversal layer lambda
        self.gradient_reversal.lambda_ = alpha
        reversed_embeddings = self.gradient_reversal(embeddings)
        domain_logits = self.domain_discriminator(reversed_embeddings)

        return {
            'embeddings': embeddings,
            'class_logits': class_logits,
            'domain_logits': domain_logits,
            'dataset_ids': torch.tensor(dataset_ids, device=embeddings.device)
        }

    def encode(self, features, dataset='iv5'):
        """
        Encode features into the shared embedding space.

        Args:
            features: Input features
            dataset: Which dataset ('iv5' or 'phmrc')

        Returns:
            Embeddings in the shared space
        """
        with torch.no_grad():
            if dataset == 'iv5':
                encoded = self.iv5_encoder(features)
            elif dataset == 'phmrc':
                encoded = self.phmrc_encoder(features)
            else:
                raise ValueError(f"Unknown dataset: {dataset}")

            embeddings = self.shared_embedding(encoded)
            return embeddings

    def predict(self, features, dataset='iv5'):
        """
        Make predictions for given features.

        Args:
            features: Input features
            dataset: Which dataset ('iv5' or 'phmrc')

        Returns:
            Class predictions (argmax of logits)
        """
        with torch.no_grad():
            embeddings = self.encode(features, dataset)
            logits = self.classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)
            return predictions


def test_dann():
    """Test DANN architecture."""
    print("Testing DANN architecture...")

    # Create model
    model = DANN(
        iv5_dim=353,
        phmrc_dim=109,
        num_classes=7,
        encoder_hidden=256,
        encoder_output=128,
        embedding_dim=64,
        dropout=0.3,
        lambda_domain=1.0
    )

    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass with IV5 data
    iv5_batch = torch.randn(16, 353)
    print(f"\nTesting IV5 forward pass...")
    output = model(iv5_features=iv5_batch, alpha=0.5)
    print(f"  Embeddings shape: {output['embeddings'].shape}")
    print(f"  Class logits shape: {output['class_logits'].shape}")
    print(f"  Domain logits shape: {output['domain_logits'].shape}")

    # Test forward pass with PHMRC data
    phmrc_batch = torch.randn(16, 109)
    print(f"\nTesting PHMRC forward pass...")
    output = model(phmrc_features=phmrc_batch, alpha=0.5)
    print(f"  Embeddings shape: {output['embeddings'].shape}")
    print(f"  Class logits shape: {output['class_logits'].shape}")
    print(f"  Domain logits shape: {output['domain_logits'].shape}")

    # Test mixed batch
    print(f"\nTesting mixed batch (IV5 + PHMRC)...")
    output = model(iv5_features=iv5_batch, phmrc_features=phmrc_batch, alpha=1.0)
    print(f"  Embeddings shape: {output['embeddings'].shape}")
    print(f"  Class logits shape: {output['class_logits'].shape}")
    print(f"  Domain logits shape: {output['domain_logits'].shape}")

    # Test encode method
    print(f"\nTesting encode method...")
    embeddings = model.encode(iv5_batch, dataset='iv5')
    print(f"  IV5 embeddings shape: {embeddings.shape}")

    embeddings = model.encode(phmrc_batch, dataset='phmrc')
    print(f"  PHMRC embeddings shape: {embeddings.shape}")

    # Test predict method
    print(f"\nTesting predict method...")
    predictions = model.predict(iv5_batch, dataset='iv5')
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions: {predictions}")

    print("\nAll tests passed!")


if __name__ == '__main__':
    test_dann()

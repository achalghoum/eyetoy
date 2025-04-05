import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR


class FeatureCollector(nn.Module):
    """Wrapper to collect intermediate features from transformer blocks"""

    def __init__(self, transformer_stack):
        super().__init__()
        self.transformer_stack = transformer_stack
        self.features = []

    def forward(self, x):
        self.features.clear()
        for transformer in self.transformer_stack.transformers:
            x = transformer(x)
            self.features.append(x)
        return x


class SegmentationEncoder(nn.Module):
    """Encoder wrapper that collects multi-scale features"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.feature_collector = FeatureCollector(encoder.transformer_stack)

        # Replace original transformer stack with feature-collecting version
        original_stack = encoder.transformer_stack
        encoder.transformer_stack = nn.Sequential(
            self.feature_collector, *[m for m in original_stack.transformers]
        )

    def forward(self, x):
        _ = self.encoder(x)  # Features collected through side effect
        return self.feature_collector.features


class UperNetDecoder(nn.Module):
    def __init__(self, encoder_channels, num_classes):
        super().__init__()
        # ... (same decoder implementation as before)


class SegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super().__init__()
        self.encoder = SegmentationEncoder(encoder)
        self.decoder = UperNetDecoder(
            encoder_channels=[
                t.out_channels for t in encoder.transformer_stack.transformers
            ],
            num_classes=num_classes,
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)

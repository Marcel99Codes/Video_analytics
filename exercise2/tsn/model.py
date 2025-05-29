import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class TSN(nn.Module):
    def __init__(self, num_classes, num_segments=4, modality='rgb', flow_init='imagenet'):
        """
        TSN model with support for RGB and Flow.

        Args:
            num_classes (int): number of output classes.
            num_segments (int): number of temporal segments.
            modality (str): 'rgb' or 'flow'.
            flow_init (str): 'imagenet' or 'random' for flow only.
        """
        super(TSN, self).__init__()
        self.num_segments = num_segments
        self.modality = modality

        if modality == 'rgb':
            # ResNet18 with ImageNet weights
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.base_model = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = resnet.fc.in_features
            self.fc = nn.Linear(self.feature_dim, num_classes)

        elif modality == 'flow':
            if flow_init == 'imagenet':
                resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif flow_init == 'random':
                resnet = resnet18(weights=None)
            else:
                raise ValueError(f"Unsupported flow_init: {flow_init}")

            original_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False)

            if flow_init == 'imagenet':
                with torch.no_grad():
                    # Average weights across 3 channels → duplicate for 10 channels
                    avg_weights = original_conv.weight.data.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    resnet.conv1.weight.data = avg_weights.repeat(1, 10, 1, 1)

            self.base_model = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_dim = resnet.fc.in_features
            self.fc = nn.Linear(self.feature_dim, num_classes)

        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def forward(self, x):
        # Expected input shape: (B, num_segments, C, H, W)
        if x.ndim != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), but got {x.shape}")

        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)  # Flatten temporal dimension into batch

        # Feature extraction
        x = self.base_model(x)  # → (B * num_segments, 512, 1, 1)
        x = x.view(x.size(0), -1)  # → (B * num_segments, 512)

        # Make sure feature_dim is set
        if not hasattr(self, "feature_dim"):
            self.feature_dim = x.shape[1]  # fallback (should be set during init)

        # Temporal average pooling
        x = x.view(b, t, self.feature_dim)  # → (B, num_segments, 512)
        x = x.mean(dim=1)  # → (B, 512)

        return self.fc(x)


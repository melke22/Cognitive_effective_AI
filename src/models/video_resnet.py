import torch.nn as nn
import torchvision.models as models

class VideoEmotionResNet(nn.Module):
    """
    ResNet18-based classifier for facial emotion frames.
    """
    def __init__(self, n_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)
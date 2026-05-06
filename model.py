import torch
import torch.nn as nn
from torchvision import models

def get_resnet_model(num_classes, pretrained=True, freeze_layers=True, dropout=0.3):
    """
    Creates a ResNet18 model for transfer learning.
    """
    if pretrained:
        # Load weights for ResNet18
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # Freeze backbone layers
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_ftrs, num_classes)
    )

    return model

if __name__ == "__main__":
    model = get_resnet_model(num_classes=38)
    print(model)
    # Verify that only the FC layer parameters are trainable
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"\nTrainable parameters: {trainable_params}")

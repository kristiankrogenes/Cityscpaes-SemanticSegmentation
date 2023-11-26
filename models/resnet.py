import torch
import torch.nn as nn
import torchvision

class ResNet(nn.Module):

    def __init__(self, backbone, num_features=None, num_classes=None):
        super(ResNet, self).__init__()
        self.backbone = backbone
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(num_features, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        segmentation_mask = self.segmentation_head(features)
        return segmentation_mask

    


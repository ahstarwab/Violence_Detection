import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from efficientnet_pytorch import EfficientNet


class MYNET(nn.Module):
    def __init__(self, sequence_size):
        super().__init__()
        self.sequence_size = sequence_size
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
        self.output_layer = nn.Linear(1000, 2)
        
    def forward(self, rgb):
        B = rgb.shape[0]
        backbone_out = self.backbone(rgb.mean(2).reshape(-1, 3, 224, 224))
        temporal_vec = self.output_layer(backbone_out).reshape(B, self.sequence_size//3, -1)
        return temporal_vec.mean(1)


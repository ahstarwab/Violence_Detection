import torch
from torch import nn
import math
import pdb
import torch.nn.functional as F

class Spatial_Attention(nn.Module):
    def __init__(self, n_length):
        super(Spatial_Attention, self).__init__()

        self.post_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=3, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3),
        )

        self.n_length = n_length

    def forward(self, x):
        '''
        x [B, T, C, H, W]
        '''
        # [ ? ? h, w]
        h, w = x.size(-2), x.size(-1)
        x = x.reshape(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1))
        # [B, T, C, (H*W)
        
        for i in range(self.n_length-1):
            d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1)
            # [B, 1, (H*W)]
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
            
        # [B, T-1, 1, (H*W)]
        diff_ = d.reshape(-1, 1, h, w)
        
        att_map = torch.nn.functional.avg_pool2d(diff_, 15, stride=1, padding=7)
        att_map = torch.nn.functional.avg_pool2d(att_map, 15, stride=1, padding=7)
        att_map = self.post_conv(att_map)
        att_map = att_map.reshape(-1, self.n_length-1, 1, h, w)
        # [B, 1, T-1, H, W]
        
        return 1 / (1 + torch.exp(-2*(att_map-0.5)))




class Temporal_Attention(nn.Module):
    def __init__(self, n_segment=10, feature_dim=196, num_class=2, dropout_ratio=0.2):

        super(Temporal_Attention, self).__init__()
        self.n_segment = n_segment
        
        self.GAP = nn.AdaptiveAvgPool1d(1)

        self.T_FC = nn.Sequential(
            nn.Linear(n_segment, n_segment//2, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(n_segment//2, n_segment, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.pred = nn.Linear(feature_dim, num_class)
        
        # fc init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.001)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        _, _, d = x.size()
        
        #Squeeze (Global Information Embedding)
        w = self.GAP(x).squeeze(2)
        #[B, new_T]

        w = self.sigmoid(self.T_FC(w))
        #[B, new_T]

        x = x * w.unsqueeze(2)
        #[B, d, new_T]
        
        x = x.sum(dim=1)
        #[B, d]
        x = self.dropout(x)
        x = self.pred(x)
        #[B, 2]
        return x

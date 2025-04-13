"""
Object recognition Things-EEG2 dataset
use 250 Hz data

The code is modified from https://github.com/eeyhsong/NICE-EEG

MUSE-Nerv-*: Use Enc_nervformer_eeg as EEG Encoder
MUSE-Nerv-GA: Need to modifify NervFormerEEGModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from torch.autograd import Variable
from einops.layers.torch import Rearrange
from einops import rearrange

from torch_geometric.nn import GATConv


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        # revised from shallownet
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        # self.instance_norm = nn.InstanceNorm2d(num_features=512)
        self.instance_norm = nn.InstanceNorm2d(num_features=1)


    def forward(self, x: Tensor) -> Tensor:
        # b, _, _, _ = x.shape
        x = self.instance_norm(x)

        x = self.tsconv(x)
        x = self.projection(x)
        return x



class STConvEEGModel(nn.Module):
    def __init__(self, output_dim):
        super(STConvEEGModel, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(num_features=1)

        self.gatnn = EEG_GAT()
        
        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (63, 1), (1, 1)),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, 40, (1, 1), stride=(1, 1)),  
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):  
        x = self.stconv(x)
        output_features = self.projection(x)

        return output_features
    

class NervFormerEEGModel(nn.Module):
    def __init__(self, output_dim):
        super(NervFormerEEGModel, self).__init__()

        self.instance_norm = nn.InstanceNorm2d(num_features=1)

        self.gatnn = EEG_GAT()

        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.stconv = nn.Sequential(
            nn.Conv2d(1, 40, (63, 1), (1, 1)),  # Spatial convolution
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (1, 25), (1, 1)),  # Temporal convolution
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.self_attn_ts = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.self_attn_st = nn.MultiheadAttention(embed_dim=40, num_heads=5)
        self.cross_attn = nn.MultiheadAttention(embed_dim=40, num_heads=8, dropout=0.75)

        self.feed_forward = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2880, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, output_dim),
        )

        self.norm1 = nn.LayerNorm(40) #d_model=40
        self.norm2 = nn.LayerNorm(40)
        self.norm3 = nn.LayerNorm(40)


    def forward(self, x):
        x = self.instance_norm(x)
        ##### Nerv-GA#####
        # x = self.gatnn(x)
        ##################
        ts_features = self.tsconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        st_features = self.stconv(x).flatten(2).permute(2, 0, 1)  # [seq_len, batch, features]
        # Attention is applied over the 250 time steps. 
        bf_ts_features, _ = self.self_attn_ts(ts_features, ts_features, ts_features)
        bf_st_features, _ = self.self_attn_st(st_features, st_features, st_features)
        # LayerNorm
        bf_ts_features = self.norm1(bf_ts_features + ts_features)
        bf_st_features = self.norm2(bf_st_features + st_features)
        combined_features = torch.cat((bf_ts_features, bf_st_features), dim=0) # need to cat?
        cf_combined_features, _ = self.cross_attn(combined_features, combined_features, combined_features)
        final_combined_features = self.norm3(cf_combined_features + combined_features)
        final_combined_features = final_combined_features.permute(1, 0, 2).flatten(1)
        output_features = self.feed_forward(final_combined_features) #[1000, 1440]

        return output_features


class EEG_GAT(nn.Module):
    def __init__(self, in_channels=250, out_channels=250):
        super(EEG_GAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = GATConv(in_channels=in_channels, out_channels=out_channels, heads=1)
        self.num_channels = 63
        # Create a list of tuples representing all possible edges between channels
        self.edge_index_list = torch.Tensor([(i, j) for i in range(self.num_channels) for j in range(self.num_channels) if i != j]).cuda()
        # Convert the list of tuples to a tensor
        self.edge_index = torch.tensor(self.edge_index_list, dtype=torch.long).t().contiguous().cuda()

    def forward(self, x):
        batch_size, _, num_channels, num_features = x.size()
        # Reshape x to (batch_size*num_channels, num_features) to pass through GATConv
        x = x.view(batch_size*num_channels, num_features)
        x = self.conv1(x, self.edge_index)
        x = x.view(batch_size, num_channels, -1)
        x = x.unsqueeze(1)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        # print("FlattenHead:: ", x.shape) #torch.Size([1000, 1440])
        return x


class Enc_nervformer_eeg(nn.Sequential):
    def __init__(self, output_dim = 1440):
        super().__init__(
            NervFormerEEGModel(output_dim),
            FlattenHead()
        )

class Enc_muse_eeg(nn.Sequential):
    def __init__(self, output_dim = 1440):
        super().__init__(
            STConvEEGModel(output_dim),
            FlattenHead()
        )
        
class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=768, drop_proj=0.5):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )


class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x 

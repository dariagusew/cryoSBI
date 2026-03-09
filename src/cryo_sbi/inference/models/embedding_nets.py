import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import math

from cryo_sbi.utils.image_utils import LowPassFilter, Mask, fourier_down_sample


EMBEDDING_NETS = {}


def add_embedding(name):
    """
    Add embedding net to EMBEDDING_NETS dict

    Args:
        name (str): name of embedding net

    Returns:
        add (function): function to add embedding net to EMBEDDING_NETS dict
    """

    def add(class_):
        EMBEDDING_NETS[name] = class_
        return class_

    return add

@add_embedding("MLP")
class MLP(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        #NN per node (bead)
        self.node_mlp = nn.Sequential( 
            nn.Linear(1, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128)
        )
        #NN pooled representation on all nodes to produce final embedds
        self.global_mlp = nn.Sequential(
            nn.Linear(128, output_dim),
            nn.LeakyReLU()
        )

    def forward(self, positions):
        """
        positions: [B, N, 3] tensor of coordinates
                   B = batch size (number of models in batch)
                   N = number of beads 
        Returns:
            [B, output_dim] embedding
        """
        #  Pairwise distances: rotationally invariant
        dists = torch.cdist(positions, positions)  # [B, N, N]
       
        #  Mean distance to other nodes
        h = dists.mean(dim=-1)  # [B, N]
        
        #  Prepare for node-level MLP: last dim = 1
        h = h.unsqueeze(-1)     # [B, N, 1]
       
        #  Node-level MLP applied per node (node level representation)
        h = self.node_mlp(h)    # [B, N, 128]
        
        #  Global pooling over nodes
        h = h.mean(dim=1)       # [B, 128]
       
        #  Global MLP to produce final embedding
        out = self.global_mlp(h)  # [B, output_dim]
       
        return out


@add_embedding('GNN')
class GNN(nn.Module):
    def __init__(self, output_dimension, hidden_dim=None, cutoff=10.0):
        super().__init__()

        # Auto-regulate: default hidden_dim to output_dimension
        if hidden_dim is None:
            hidden_dim = output_dimension

        self.cutoff = cutoff
        self.hidden_dim = hidden_dim
        self.output_dimension = output_dimension

        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dimension)
        )

    @staticmethod
    def batch_coords_to_graph(coords_batch, cutoff=10):
        """
        Fully vectorized graph construction.
        coords_batch: [B, N, 3]
        Returns:
            x: [B*N,1] node features
            edge_index: [2, total_edges]
            edge_attr: [total_edges,1]
            batch: [B*N] node-to-graph mapping
        """
        B, N, _ = coords_batch.shape
        device = coords_batch.device

        # Node features
        x = torch.ones(B * N, 1, device=device)
        batch = torch.arange(B, device=device).repeat_interleave(N)

        # Compute all pairwise distances per graph using broadcasting
        coords_i = coords_batch.unsqueeze(2)  # [B,N,1,3]
        coords_j = coords_batch.unsqueeze(1)  # [B,1,N,3]
        dist = torch.norm(coords_i - coords_j, dim=-1)  # [B,N,N]

        # Mask: no self-loops and within cutoff
        mask = (dist < cutoff) & (~torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0))
        # Get edge indices and attributes
        b_idx, i_idx, j_idx = mask.nonzero(as_tuple=True)  # b, src, tgt
        edge_index = torch.stack([i_idx + b_idx * N, j_idx + b_idx * N], dim=0)  # [2, total_edges]
        edge_attr = dist[b_idx, i_idx, j_idx].unsqueeze(-1)  # [total_edges,1]

        return x, edge_index, edge_attr, batch

    def forward(self, coords_batch):
        B, N, _ = coords_batch.shape
        device = coords_batch.device

        # Build graph using instance cutoff
        x, edge_index, edge_attr, batch = self.batch_coords_to_graph(coords_batch, self.cutoff)
        
        # Edge MLP
        messages = self.edge_mlp(edge_attr)  # [total_edges, hidden_dim]

        # Aggregate messages per node
        target_nodes = edge_index[1]
        node_messages = torch.zeros((B*N, messages.shape[1]), device=device)
        counts = torch.zeros(B*N, device=device)
        node_messages = node_messages.index_add(0, target_nodes, messages)
        counts = counts.index_add(0, target_nodes, torch.ones_like(target_nodes, dtype=torch.float))
        counts = counts.clamp(min=1.0).unsqueeze(-1)
        node_features = node_messages / counts

        # Node MLP
        node_embeddings = self.node_mlp(node_features)  # [B*N, output_dimension]

        # Graph-level mean pooling
        graph_embeddings = torch.zeros((B, node_embeddings.shape[1]), device=device)
        graph_embeddings = graph_embeddings.index_add(0, batch, node_embeddings)
        counts_graph = torch.zeros(B, device=device)
        counts_graph = counts_graph.index_add(0, batch, torch.ones(B*N, device=device))
        counts_graph = counts_graph.clamp(min=1.0).unsqueeze(-1)
        graph_embeddings = graph_embeddings / counts_graph

        return graph_embeddings

@add_embedding("RESNET18")
class ResNet18_Encoder(nn.Module):
    def __init__(self, output_dimension: int, D: int = 128):
        super(ResNet18_Encoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET50")
class ResNet50_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet50_Encoder, self).__init__()

        self.resnet = models.resnet50()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("RESNET101")
class ResNet101_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet101_Encoder, self).__init__()

        self.resnet = models.resnet101()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("CONVNET")
class ConvNet_Encoder(nn.Module):
    def __init__(self, output_dimension: int, D: int = 128):
        super(ConvNet_Encoder, self).__init__()

        self.convnet = models.convnext_tiny()
        self.convnet.features[0][0] = nn.Conv2d(
            1, 96, kernel_size=(4, 4), stride=(4, 4)
        )
        self.convnet.classifier[2] = nn.Linear(
            in_features=768, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.convnet(x)
        return x


@add_embedding("CONVNET_2")
class RegNetX_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(RegNetX_Encoder, self).__init__()

        self.regnetx = models.regnet_x_3_2gf()
        self.regnetx.stem[0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.regnetx.fc = nn.Linear(
            in_features=1008, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.regnetx(x)
        return x


@add_embedding("EFFICIENT")
class EfficientNet_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(EfficientNet_Encoder, self).__init__()

        self.efficient_net = models.efficientnet_b3().features
        self.efficient_net[0][0] = nn.Conv2d(
            1, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.leakyrelu = nn.LeakyReLU()
        self.linear = nn.Linear(1536, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.efficient_net(x)
        x = self.avg_pool(x).flatten(start_dim=1)
        x = self.leakyrelu(self.linear(x))
        return x


@add_embedding("SWINS")
class SwinTransformerS_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(SwinTransformerS_Encoder, self).__init__()

        self.swin_transformer = models.swin_t()
        self.swin_transformer.features[0][0] = nn.Conv2d(
            1, 96, kernel_size=(4, 4), stride=(4, 4)
        )
        self.swin_transformer.head = nn.Linear(
            in_features=768, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.swin_transformer(x)
        return x


@add_embedding("WIDERES50")
class WideResnet50_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(WideResnet50_Encoder, self).__init__()

        self.wideresnet = models.wide_resnet50_2()
        self.wideresnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.wideresnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("WIDERES101")
class WideResnet101_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(WideResnet101_Encoder, self).__init__()

        self.wideresnet = models.wide_resnet101_2()
        self.wideresnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.linear = nn.Linear(1000, output_dimension)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.wideresnet(x)
        x = self.linear(nn.functional.relu(x))
        return x


@add_embedding("REGNETY")
class RegNetY_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(RegNetY_Encoder, self).__init__()

        self.regnety = models.regnet_y_1_6gf()
        self.regnety.stem[0] = nn.Conv2d(
            1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.regnety.fc = nn.Linear(
            in_features=888, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.regnety(x)
        return x


@add_embedding("SHUFFLENET")
class ShuffleNet_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ShuffleNet_Encoder, self).__init__()

        self.shuffle_net = models.shufflenet_v2_x0_5()
        self.shuffle_net.conv1[0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )
        self.shuffle_net.fc = nn.Linear(
            in_features=1024, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.shuffle_net(x)
        return x


@add_embedding("RESNET18_FFT_FILTER")
class ResNet18_FFT_Encoder(nn.Module):
    def __init__(self, output_dimension: int, D: int = 128):
        super(ResNet18_FFT_Encoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(D, 25)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET18_FFT_FILTER_132")
class ResNet18_FFT_Encoder_132(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder_132, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(132, 25)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET18_FFT_FILTER_224")
class ResNet18_FFT_Encoder_224(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder_224, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(224, 25)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET18_FFT_FILTER_256")
class ResNet18_FFT_Encoder_256(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder_256, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(256, 10)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET34")
class ResNet34_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet34_Encoder, self).__init__()
        self.resnet = models.resnet34()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("RESNET34_FFT_FILTER_256")
class ResNet34_Encoder_FFT_FILTER_256(nn.Module):
    def __init__(self, output_dimension: int):
        super(ResNet34_Encoder_FFT_FILTER_256, self).__init__()
        self.resnet = models.resnet34()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )
        self._fft_filter = LowPassFilter(256, 50)

    def forward(self, x):
        # Low pass filter images
        x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.resnet(x)
        return x


@add_embedding("VGG19")
class VGG19_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(VGG19_Encoder, self).__init__()

        self.vgg19 = models.vgg19_bn().features
        self.vgg19[0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.feedforward = nn.Sequential(
            *[
                nn.Linear(in_features=25088, out_features=4096),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=4096, out_features=output_dimension, bias=True),
                nn.ReLU(inplace=True),
            ]
        )

    def forward(self, x):
        # Low pass filter images
        # x = self._fft_filter(x)
        # Proceed as normal
        x = x.unsqueeze(1)
        x = self.vgg19(x)
        x = self.avgpool(x).flatten(start_dim=1)
        x = self.feedforward(x)
        return x


@add_embedding("ConvEncoder_Tutorial")
class ConvEncoder(nn.Module):
    def __init__(self, output_dimension: int):
        super(ConvEncoder, self).__init__()
        ndf = 16  # fixed for the tutorial
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, output_dimension, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.main(x)
        return x.view(x.size(0), -1)  # flatten


@add_embedding("SPATIAL_CRYO")
class SpatialCryoEncoder(nn.Module):
    """
    Lightweight spatial encoder for cryo-EM images
    Inspired by CryoDRGN's ConvEncoder but flexible for any image size
    
    Architecture:
    - All-convolutional design (no heavy FC layers)
    - Progressive downsampling: D → D/2 → ... → 4 → 1
    - Channel progression: 1 → 16 → 32 → 64 → ... → output_dim
    - Final conv trick: 4x4 → 1x1 instead of flatten+FC
    
    Parameters:
    - D=64:  ~300K  (CryoDRGN scale)
    - D=128: ~1.75M (6x lighter than ResNet18)
    - D=256: ~7M    (still lighter than ResNet18)
    """
    def __init__(self, output_dimension: int, D: int = 128):
        super(SpatialCryoEncoder, self).__init__()
        
        self.D = D
        self.output_dimension = output_dimension
        
        # Base channel dimension (CryoDRGN choice)
        ndf = 16
        
        # Calculate downsampling stages: D → 4
        n_stages = int(math.log2(D)) - 2
        
        if n_stages < 1:
            raise ValueError(f"Image size D={D} too small. Minimum D=8.")
        
        layers = []
        in_channels = 1
        
        # Progressive downsampling with exponential channel growth
        for i in range(n_stages):
            out_channels = ndf * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        # Final convolutional layer: 4x4 → 1x1
        # This eliminates the need for large FC layers
        layers.append(
            nn.Conv2d(in_channels, output_dimension,
                     kernel_size=4, stride=1, padding=0, bias=False)
        )
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # Output normalization for stable flow training
        self.output_norm = nn.LayerNorm(output_dimension)
    
    def forward(self, x):
        """
        Args:
            x: [B, D, D] or [B, 1, D, D] images
        
        Returns:
            embeddings: [B, output_dimension]
        """
        # Ensure 4D input [B, 1, D, D]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Convolutional encoding
        x = self.conv_encoder(x)  # [B, output_dim, 1, 1]
        
        # Flatten to [B, output_dim]
        x = x.view(x.size(0), -1)
        
        # Normalize for flow training
        x = self.output_norm(x)
        
        return x

@add_embedding("SPATIAL_CRYO_FFT_FILTER")
class SpatialCryoFFTEncoder(nn.Module):
    """
    Lightweight spatial encoder with FFT preprocessing for noisy cryo-EM images
    Combines SPATIAL_CRYO's efficient architecture with FFT low-pass filtering
    
    Architecture:
    - FFT low-pass filter preprocessing (removes high-frequency noise)
    - All-convolutional design (no heavy FC layers)
    - Progressive downsampling: D → D/2 → ... → 4 → 1
    - Channel progression: 1 → 16 → 32 → 64 → ... → output_dim
    - Final conv trick: 4x4 → 1x1 instead of flatten+FC
    - LayerNorm output for stable flow training
    
    Parameters:
    - D=128: ~1.75M params (default, optimized for 128x128 images)
    - Cutoff=25: FFT filter cutoff frequency (same as RESNET18_FFT_FILTER)
    """
    def __init__(self, output_dimension: int, D: int = 128):
        super(SpatialCryoFFTEncoder, self).__init__()
        
        self.D = D
        self.output_dimension = output_dimension
        
        # FFT low-pass filter (matches RESNET18_FFT_FILTER for D=128)
        self._fft_filter = LowPassFilter(D, 25)
        
        # Base channel dimension (CryoDRGN choice)
        ndf = 16
        
        # Calculate downsampling stages: D → 4
        n_stages = int(math.log2(D)) - 2
        
        if n_stages < 1:
            raise ValueError(f"Image size D={D} too small. Minimum D=8.")
        
        layers = []
        in_channels = 1
        
        # Progressive downsampling with exponential channel growth
        for i in range(n_stages):
            out_channels = ndf * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        # Final convolutional layer: 4x4 → 1x1
        layers.append(
            nn.Conv2d(in_channels, output_dimension,
                     kernel_size=4, stride=1, padding=0, bias=False)
        )
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # Output normalization for stable flow training
        self.output_norm = nn.LayerNorm(output_dimension)
    
    def forward(self, x):
        """
        Args:
            x: [B, D, D] or [B, 1, D, D] images
        
        Returns:
            embeddings: [B, output_dimension]
        """
        # Apply FFT low-pass filter (removes high-frequency noise)
        x = self._fft_filter(x)
        
        # Ensure 4D input [B, 1, D, D]
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        # Convolutional encoding
        x = self.conv_encoder(x)  # [B, output_dim, 1, 1]
        
        # Flatten to [B, output_dim]
        x = x.view(x.size(0), -1)
        
        # Normalize for flow training
        x = self.output_norm(x)
        
        return x

@add_embedding("SPATIAL_CRYO_GAUSS_FFT_FILTER")
class SpatialCryoGaussFFTEncoder(nn.Module):
    """
    Lightweight spatial encoder with a self-contained, learnable Gaussian FFT filter.
    
    This version encapsulates all filtering logic, removing the dependency on an
    external filter class.
    
    Architecture:
    - Internal Gaussian low-pass filter (sigma is learned during training)
    - All-convolutional design (no heavy FC layers)
    - Progressive downsampling: D → D/2 → ... → 4 → 1
    - Channel progression: 1 → 16 → 32 → 64 → ... → output_dim
    - Final conv trick: 4x4 → 1x1 instead of flatten+FC
    - LayerNorm output for stable flow training
    
    Parameters:
    - output_dimension (int): The dimension of the latent embedding.
    - D (int): The side length of the input image (e.g., 128).
    - initial_sigma (float): The starting value for the filter's sigma.
    """
    def __init__(self, output_dimension: int, D: int = 128, initial_sigma: float = 25.0):
        super(SpatialCryoGaussFFTEncoder, self).__init__()
        
        self.D = D
        self.output_dimension = output_dimension
        
        # 1. Create a learnable parameter for sigma.
        # We learn log(sigma) to ensure sigma is always positive.
        self.log_sigma = nn.Parameter(torch.tensor(math.log(initial_sigma)))
        
        # 2. Pre-calculate the grid of squared radii, which is constant.
        # This grid is used to generate the Gaussian mask in the forward pass.
        grid = torch.linspace(-0.5 * (D - 1), 0.5 * (D - 1), D)
        r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
        
        # 3. Register r_2d as a buffer. This makes it part of the model's state
        # (e.g., moves to GPU with .to(device)) but not a learnable parameter.
        self.register_buffer('_r_2d', r_2d)
        
        # Base channel dimension (CryoDRGN choice)
        ndf = 16
        
        # Calculate downsampling stages: D → 4
        n_stages = int(math.log2(D)) - 2
        
        if n_stages < 1:
            raise ValueError(f"Image size D={D} too small. Minimum D=8.")
        
        layers = []
        in_channels = 1
        
        # Progressive downsampling with exponential channel growth
        for i in range(n_stages):
            out_channels = ndf * (2 ** i)
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ])
            in_channels = out_channels
        
        # Final convolutional layer: 4x4 → 1x1
        layers.append(
            nn.Conv2d(in_channels, output_dimension,
                     kernel_size=4, stride=1, padding=0, bias=False)
        )
        
        self.conv_encoder = nn.Sequential(*layers)
        
        # Output normalization for stable flow training
        self.output_norm = nn.LayerNorm(output_dimension)

    def _apply_fft_filter(self, image: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Internal method to apply the low-pass filter using the current sigma.
        """
        # Calculate the Gaussian mask dynamically using the current sigma
        # Add a small epsilon for numerical stability if sigma is near zero
        mask = torch.exp(-self._r_2d / (2 * sigma**2 + 1e-8))

        # Apply FFT. Use dim=(-2, -1) for robustness with different input shapes.
        fft_image = torch.fft.fft2(image, dim=(-2, -1))
        fft_image = torch.fft.fftshift(fft_image, dim=(-2, -1))

        # Reshape the mask to broadcast correctly with the image tensor
        # (e.g., from [H, W] to [1, 1, H, W] for a [B, C, H, W] image)
        broadcast_shape = [1] * (image.dim() - 2) + list(mask.shape)
        fft_image = fft_image * mask.view(*broadcast_shape)

        # Invert FFT
        fft_image = torch.fft.ifftshift(fft_image, dim=(-2, -1))
        reconstructed = torch.fft.ifft2(fft_image, dim=(-2, -1)).real
        return reconstructed

    def forward(self, x):
        """
        Args:
            x: [B, D, D] or [B, 1, D, D] images
        
        Returns:
            embeddings: [B, output_dimension]
        """
        # Get the positive sigma value from its learned logarithm
        sigma = torch.exp(self.log_sigma)
        
        # Apply the internal FFT low-pass filter with the learned sigma
        x = self._apply_fft_filter(x, sigma)
        
        # Ensure 4D input [B, 1, D, D] for the convolutional layers
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional encoding
        x = self.conv_encoder(x)  # [B, output_dim, 1, 1]
        
        # Flatten to [B, output_dim]
        x = x.view(x.size(0), -1)
        
        # Normalize for flow training
        x = self.output_norm(x)
        
        return x
    
# @add_embedding("HETSIREN")
# class HETSIREN(torch.Module):
#     def __init__(self, output_dimension: int, D: int = 128, architecture="convnn",
#                  downsample = False,
#                  trainable = True
#                  ):
#         super(HETSIREN, self).__init__()
#         #HETSIREN ENCODER 

#         self.D = D
#         self.output_dimension = output_dimension
#         self.downsample = downsample
#         self.trainable = trainable 
#         self.n_pixels = n_pixels

#     def forward(self, x):

#         if x.dim() == 3:
#             x = x.unsqueeze(1)

#         if self.downsample:
#             x = fourier_down_sample(x, self.D, self.n_pixels)
#             x = x.view(-1, 1, 64, 64)
#             x = nn.ReLU()(x)
#         else:
#             x = nn.Flatten()(x)
#             x = nn.Dense(64 * 64)(x)
#             x = x.view(-1, 1, 64, 64)


#         # First conv: 4 filters, 5x5 kernel, stride 2, padding="same"
#         conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, stride=2, padding=2)
#         x = nn.ReLU(conv1(x))  # activation applied separately

#         # Second conv: 8 filters, 5x5 kernel, stride 2, padding="same"
#         conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2)
#         b1_out = nn.ReLU(conv2(x))

#         # Third conv branch
#         conv3a = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)  # "same" padding for 1x1 = 0
#         b2_x = nn.ReLU(conv3a(b1_out))

#         conv3b = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, padding=0)
#         b2_x = conv3b(b2_x)  # linear activation → no ReLU
#         # b1_out and b2_x: tensors of same shape
#         b2_add = b1_out + b2_x  # element-wise addition
#         b2_add = nn.ReLU(b2_add)  # ReLU activation

#         conv1 = nn.Conv2d(1, 4, 5, stride=2, padding=2)
#         conv2 = nn.Conv2d(4, 8, 5, stride=2, padding=2)

#         x = nn.ReLU(conv1(x))
#         b1_out = nn.ReLU(conv2(x))

#         # ---- Block 2: residual branch ----
#         conv_b2_1 = nn.Conv2d(8, 8, 1, stride=1, padding=0)  # linear activation
#         b2_x = conv_b2_1(b1_out)
#         b2_add = nn.ReLU(b1_out + b2_x)  # residual add + ReLU

#         # Another 1x1 linear residual (loop of 1)
#         conv_b2_loop = nn.Conv2d(8, 8, 1, stride=1, padding=0)
#         b2_x = conv_b2_loop(b2_add)
#         b2_add = nn.ReLU(b2_add + b2_x)

#         # Downsample after residual
#         conv_b2_down = nn.Conv2d(8, 16, 3, stride=2, padding=1)
#         b2_out = nn.ReLU(conv_b2_down(b2_add))

#         # ---- Block 3: residual branch ----
#         conv_b3_1 = nn.Conv2d(16, 16, 1, stride=1, padding=0)  # ReLU
#         conv_b3_2 = nn.Conv2d(16, 16, 1, stride=1, padding=0)  # linear

#         b3_x = nn.ReLU(conv_b3_1(b2_out))
#         b3_x = conv_b3_2(b3_x)
#         b3_add = nn.ReLU(b2_out + b3_x)

#         # Another 1x1 linear residual (loop of 1)
#         conv_b3_loop = nn.Conv2d(16, 16, 1, stride=1, padding=0)
#         b3_x = conv_b3_loop(b3_add)
#         b3_add = nn.ReLU(b3_add + b3_x)

#         # Downsample after residual
#         conv_b3_down = nn.Conv2d(16, 16, 3, stride=2, padding=1)
#         b3_out = nn.ReLU(conv_b3_down(b3_add))

#         # ---- Flatten and Dense layers ----
#         x = torch.flatten(b3_out, start_dim=1)  # flatten all except batch

#         # Fully connected layers (4 layers of 256 neurons with ReLU)
#         fc1 = nn.Linear(x.shape[1], 256)
#         fc2 = nn.Linear(256, 256)
#         fc3 = nn.Linear(256, 256)
#         fc4 = nn.Linear(256, 256)

#         x = nn.ReLU(fc1(x))
#         x = nn.ReLU(fc2(x))
#         x = nn.ReLU(fc3(x))
#         x = nn.ReLU(fc4(x))

#         return x






class Residual1x1Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        x = x + out
        return self.relu(x)


@add_embedding("HETSIREN")
class HETSIREN(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 D=64,
                 architecture="convnn",
                 refPose=True,
                 mode="spa",
                 downsample=False):
        super(HETSIREN, self).__init__()

        self.embedding_dim = embedding_dim
        self.D = D
        self.mode = mode
        self.architecture = architecture
        self.downsample = downsample

        # ====================================================
        # CONVNN
        # ====================================================
        if architecture == "convnn":
            if downsample:
                self.pre_fc = None
            else:
                self.pre_fc = nn.Linear(D * D, 64 * 64)

            self.conv1 = nn.Conv2d(1, 4, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(4, 8, 5, stride=2, padding=2)

            self.b2_conv1 = nn.Conv2d(8, 8, 1)
            self.b2_conv2 = nn.Conv2d(8, 8, 1)
            self.b2_res = Residual1x1Block(8)

            self.b2_out = nn.Conv2d(8, 16, 3, stride=2, padding=1)

            self.b3_conv1 = nn.Conv2d(16, 16, 1)
            self.b3_conv2 = nn.Conv2d(16, 16, 1)
            self.b3_res = Residual1x1Block(16)

            self.b3_out = nn.Conv2d(16, 16, 3, stride=2, padding=1)

            self.fc_layers = nn.ModuleList(
                [nn.Linear(16 * 4 * 4, embedding_dim)] +
                [nn.Linear(embedding_dim, embedding_dim) for _ in range(3)]
            )

        # ====================================================
        # DEEPCONV
        # ====================================================
        elif architecture == "deepconv":
            self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)

            self.deep_res2 = nn.ModuleList([Residual1x1Block(128) for _ in range(12)])
            self.b2_out = nn.Conv2d(128, 256, 3, stride=2, padding=1)

            self.deep_res3 = nn.ModuleList([Residual1x1Block(256) for _ in range(12)])
            self.b3_out = nn.Conv2d(256, 512, 3, stride=2, padding=1)

            self.fc1 = nn.Linear(512 * 4 * 4, embedding_dim)
            self.fc_res = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim) for _ in range(4)])

        # ====================================================
        # MLP
        # ====================================================
        elif architecture == "mlpnn":
            self.mlp = nn.Sequential(
                nn.Linear(D * D, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, embedding_dim),
                nn.ReLU()
            )

        # ====================================================
        # LATENT HEAD
        # ====================================================
        self.latent_layers = nn.ModuleList(
            [nn.Linear(embedding_dim if architecture != "deepconv" else embedding_dim, embedding_dim)] +
            [nn.Linear(embedding_dim, embedding_dim) for _ in range(2)]
        )

        # ====================================================
        # ROWS HEAD
        # ====================================================
        self.rows_layers = nn.ModuleList(
            [nn.Linear(embedding_dim if architecture != "deepconv" else embedding_dim, 256)] +
            [nn.Linear(256, 256) for _ in range(2)]
        )

        # ====================================================
        # SHIFTS HEAD
        # ====================================================
        self.shifts_layers = nn.ModuleList(
            [nn.Linear(embedding_dim if architecture != "deepconv" else embedding_dim, 256)] +
            [nn.Linear(256, 256) for _ in range(2)]
        )

        # Freeze if refPose=False
        if not refPose:
            for p in self.rows_layers.parameters():
                p.requires_grad = False
            for p in self.shifts_layers.parameters():
                p.requires_grad = False

        # ====================================================
        # TOMO LABEL HEAD
        # ====================================================
        if mode == "tomo":
            self.label_layers = nn.ModuleList(
                [nn.Linear(100, 1024)] +
                [nn.Linear(1024, 1024) for _ in range(2)] +
                [nn.Linear(1024, 256) for _ in range(2)]
            )
            
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel

        batch_size = x.size(0)
        x = F.interpolate(x, size=(self.D, self.D))  # make input size flexible

        # ================= CONVNN =================
        if self.architecture == "convnn":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            b2 = F.relu(self.b2_conv1(x))
            b2 = self.b2_conv2(b2)
            x = F.relu(x + b2)
            x = self.b2_res(x)
            x = F.relu(self.b2_out(x))

            b3 = F.relu(self.b3_conv1(x))
            b3 = self.b3_conv2(b3)
            x = F.relu(x + b3)
            x = self.b3_res(x)
            x = F.relu(self.b3_out(x))

            # Adaptive pooling to fixed size (e.g., 4x4)
            x = F.adaptive_avg_pool2d(x, (4, 4))
            x = torch.flatten(x, 1)

            # FC to embedding_dim
            for layer in self.fc_layers:
                x = F.relu(layer(x))

        # ================= DEEPCONV =================
        elif self.architecture == "deepconv":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            for block in self.deep_res2:
                x = block(x)
            x = F.relu(self.b2_out(x))
            for block in self.deep_res3:
                x = block(x)
            x = F.relu(self.b3_out(x))
            x = F.adaptive_avg_pool2d(x, (4,4))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            for layer in self.fc_res:
                x = F.relu(layer(x)) + x

        # ================= MLP =================
        elif self.architecture == "mlpnn":
            x = x.view(batch_size, -1)
            x = self.mlp(x)

        # LATENT mapping
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.fc_layers) - 1:
                x = F.relu(x)
            return x



# Residual 1x1 Block (linear activation after first conv, as in Keras)
class Residual1x1Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)  # linear activation in Keras
        return F.relu(out + x)

@add_embedding("HETSIREN2")
class HETSIREN2(nn.Module):
    def __init__(self, output_dimension=256, D=128, architecture="convnn", mode="spa", refPose=True):
        """
        output_dimension: latent embedding size
        D: side length of input image
        """
        super(HETSIREN2, self).__init__()
        self.architecture = architecture
        self.mode = mode
        self.D = D
        self.output_dimension = output_dimension

        # ================= CONVNN =================
        if architecture == "convnn":
            self.pre_fc = nn.Linear(D * D, 64 * 64)  # Keras: flatten → Dense → reshape

            self.conv1 = nn.Conv2d(1, 4, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(4, 8, 5, stride=2, padding=2)

            self.b2_conv1 = nn.Conv2d(8, 8, 1)
            self.b2_conv2 = nn.Conv2d(8, 8, 1)
            self.b2_res = Residual1x1Block(8)
            self.b2_out = nn.Conv2d(8, 16, 3, stride=2, padding=1)

            self.b3_conv1 = nn.Conv2d(16, 16, 1)
            self.b3_conv2 = nn.Conv2d(16, 16, 1)
            self.b3_res = Residual1x1Block(16)
            self.b3_out = nn.Conv2d(16, 16, 3, stride=2, padding=1)

            self.fc_layers = nn.ModuleList([nn.Linear(16*4*4, output_dimension)] +
                                           [nn.Linear(output_dimension, output_dimension) for _ in range(2)])

        # ================= DEEPCONV =================
        elif architecture == "deepconv":
            self.conv1 = nn.Conv2d(1, 64, 5, stride=2, padding=2)
            self.conv2 = nn.Conv2d(64, 128, 5, stride=2, padding=2)

            self.deep_res2 = nn.ModuleList([Residual1x1Block(128) for _ in range(12)])
            self.b2_out = nn.Conv2d(128, 256, 3, stride=2, padding=1)

            self.deep_res3 = nn.ModuleList([Residual1x1Block(256) for _ in range(12)])
            self.b3_out = nn.Conv2d(256, 512, 3, stride=2, padding=1)

            self.fc1 = nn.Linear(512*4*4, 512)
            self.fc_res = nn.ModuleList([nn.Linear(512, 512) for _ in range(4)])

            self.latent_fc = nn.ModuleList([nn.Linear(512, output_dimension)] +
                                           [nn.Linear(output_dimension, output_dimension) for _ in range(2)])

        # ================= MLP =================
        elif architecture == "mlpnn":
            self.mlp = nn.Sequential(
                nn.Linear(D * D, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU()
            )
            self.latent_fc = nn.ModuleList([nn.Linear(1024, output_dimension)] +
                                           [nn.Linear(output_dimension, output_dimension) for _ in range(2)])
        self.output_norm = nn.LayerNorm(output_dimension)

    def forward(self, x):
        batch_size = x.size(0)
        if x.dim() == 3:
            x = x.unsqueeze(1)  # add channel

        # ================= CONVNN =================
        if self.architecture == "convnn":
            # Keras: Flatten → Dense → reshape
            x = x.view(batch_size, -1)
            x = self.pre_fc(x)
            x = x.view(batch_size, 1, 64, 64)

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))

            b2 = F.relu(self.b2_conv1(x))
            b2 = self.b2_conv2(b2)
            x = F.relu(x + b2)
            x = self.b2_res(x)
            x = F.relu(self.b2_out(x))

            b3 = F.relu(self.b3_conv1(x))
            b3 = self.b3_conv2(b3)
            x = F.relu(x + b3)
            x = self.b3_res(x)
            x = F.relu(self.b3_out(x))

            x = torch.flatten(x, 1)  # Keras: Flatten

            for i, layer in enumerate(self.fc_layers):
                x = layer(x)
                if i < len(self.fc_layers)-1:
                    x = F.relu(x)

        # ================= DEEPCONV =================
        elif self.architecture == "deepconv":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            for block in self.deep_res2:
                x = block(x)
            x = F.relu(self.b2_out(x))
            for block in self.deep_res3:
                x = block(x)
            x = F.relu(self.b3_out(x))
            x = torch.flatten(x, 1)
            x = F.relu(self.fc1(x))
            for layer in self.fc_res:
                x = F.relu(layer(x)) + x

            for i, layer in enumerate(self.latent_fc):
                x = layer(x)
                if i < len(self.latent_fc)-1:
                    x = F.relu(x)

        # ================= MLP =================
        elif self.architecture == "mlpnn":
            x = x.view(batch_size, -1)
            x = self.mlp(x)

            for i, layer in enumerate(self.latent_fc):
                x = layer(x)
                if i < len(self.latent_fc)-1:
                    x = F.relu(x)
        x = self.output_norm(x)
        return x
       
    # def forward(self, x):

    #     if x.dim() == 3:
    #         x = x.unsqueeze(1)  # add channel dimension

    #     batch_size = x.size(0)
    #     x = F.interpolate(x, size=(self.D, self.D))

    #     # ================= CONVNN =================
    #     if self.architecture == "convnn":
    #         if not self.downsample:
    #             x = x.view(batch_size, -1)
    #             x = self.pre_fc(x)
    #             x = x.view(batch_size, 1, 64, 64)
    #         else:
    #             x = F.interpolate(x, size=(64, 64))

    #         x = F.relu(self.conv1(x))
    #         x = F.relu(self.conv2(x))

    #         b2 = F.relu(self.b2_conv1(x))
    #         b2 = self.b2_conv2(b2)
    #         x = F.relu(x + b2)
    #         x = self.b2_res(x)

    #         x = F.relu(self.b2_out(x))

    #         b3 = F.relu(self.b3_conv1(x))
    #         b3 = self.b3_conv2(b3)
    #         x = F.relu(x + b3)
    #         x = self.b3_res(x)

    #         x = F.relu(self.b3_out(x))
    #         x = torch.flatten(x, 1)

    #         for layer in self.fc_layers:
    #             x = F.relu(layer(x))

    #     # ================= DEEPCONV =================
    #     elif self.architecture == "deepconv":
    #         x = F.relu(self.conv1(x))
    #         x = F.relu(self.conv2(x))
    #         for block in self.deep_res2:
    #             x = block(x)
    #         x = F.relu(self.b2_out(x))
    #         for block in self.deep_res3:
    #             x = block(x)
    #         x = F.relu(self.b3_out(x))
    #         x = torch.flatten(x, 1)
    #         x = F.relu(self.fc1(x))
    #         for layer in self.fc_res:
    #             aux = F.relu(layer(x))
    #             x = x + aux

    #     # ================= MLP =================
    #     elif self.architecture == "mlpnn":
    #         x = x.view(batch_size, -1)
    #         x = self.mlp(x)

        
    #     for layer in self.latent_layers:
    #         x = F.relu(layer(x))
    #     return x



if __name__ == "__main__":
    pass

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from cryo_sbi.utils.image_utils import LowPassFilter, Mask


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
    def __init__(self, in_dim=1, hidden_dim=64, output_dimension=128):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        #print(self.edge_mlp)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dimension)
        )

    @staticmethod
    def batch_coords_to_graph(coords_batch, cutoff=5):
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

    def forward(self, coords_batch, cutoff=5):
        B, N, _ = coords_batch.shape
        #print('B, N, _',B, N, _)
        device = coords_batch.device

        # Build graph
        x, edge_index, edge_attr, batch = self.batch_coords_to_graph(coords_batch, cutoff)
        
        #print('x, edge_index, edge_attr, batch',x.shape, edge_index.shape, edge_attr.shape, batch.shape)
        #print(edge_attr.shape)  # [num_edges, 1]
        # Edge MLP
        messages = self.edge_mlp(edge_attr)  # [total_edges, hidden_dim]
        #print('messages',messages.shape)

        # Aggregate messages per node
        target_nodes = edge_index[1]
        node_messages = torch.zeros((B*N, messages.shape[1]), device=device)
        counts = torch.zeros(B*N, device=device)
        node_messages = node_messages.index_add(0, target_nodes, messages)
        counts = counts.index_add(0, target_nodes, torch.ones_like(target_nodes, dtype=torch.float))
        counts = counts.clamp(min=1.0).unsqueeze(-1)
        node_features = node_messages / counts
        #print(node_features.shape)

        # Node MLP
        node_embeddings = self.node_mlp(node_features)  # [B*N, out_dim]
        #print('node_embeddings',node_embeddings.shape)
        # Graph-level mean pooling
        graph_embeddings = torch.zeros((B, node_embeddings.shape[1]), device=device)
        graph_embeddings = graph_embeddings.index_add(0, batch, node_embeddings)
        counts_graph = torch.zeros(B, device=device)
        counts_graph = counts_graph.index_add(0, batch, torch.ones(B*N, device=device))
        counts_graph = counts_graph.clamp(min=1.0).unsqueeze(-1)
        graph_embeddings = graph_embeddings / counts_graph

        #print('graph_embeddings',graph_embeddings.shape)
        #print(edge_attr.shape)  # [num_edges, 1]
        #print(self.edge_mlp[0].in_features)  # should be 1

        return graph_embeddings

@add_embedding("RESNET18")
class ResNet18_Encoder(nn.Module):
    def __init__(self, output_dimension: int):
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
    def __init__(self, output_dimension: int):
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


@add_embedding("CONVNET")
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
    def __init__(self, output_dimension: int):
        super(ResNet18_FFT_Encoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet.fc = nn.Linear(
            in_features=512, out_features=output_dimension, bias=True
        )

        self._fft_filter = LowPassFilter(128, 25)

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


if __name__ == "__main__":
    pass

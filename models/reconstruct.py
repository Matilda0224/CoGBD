import torch
import torch.nn as nn
import torch.optim as optim

# Define the Autoencoder class
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        # Encoder: MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size // 3),
            nn.ReLU(True)
        )
        # Decoder: MLP
        self.decoder = nn.Sequential(
            nn.Linear(input_size // 3, 2 * input_size // 3),
            nn.ReLU(True),
            nn.Linear(2 * input_size // 3, input_size),
            nn.Sigmoid() # Use Sigmoid if the input data is normalized between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MLPAE(nn.Module):
    # def __init__(self, ori_x, trigger, device, epochs):
    def __init__(self, ori_x, device, epochs):
        super(MLPAE, self).__init__()
        self.device = device
        self.model = Autoencoder(len(ori_x[0])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.epochs = epochs
        self.ori_x = ori_x
        # self.trigger = triggers

    def fit(self):
        for epoch in range(self.epochs):
            output = self.model(self.ori_x)
            loss = self.criterion(output, self.ori_x)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # if (epoch + 1) % 10 == 0:
            #     print(f"[AE][{epoch+1}/{self.epochs}] loss={loss:.4f}")

    # def inference(self, input):
    #     self.model.eval()
    #     reconstruction_errors = []
    #     with torch.no_grad():
    #         for sample in input:
    #             reconstructed = self.model(sample)
    #             loss = self.criterion(reconstructed, sample)
    #             reconstruction_errors.append(loss.item())
    #     return reconstruction_errors
    def inference(self, input):
        self.model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for sample in input:
                reconstructed = self.model(sample)
                loss = self.criterion(reconstructed, sample) #能否加入什么更好的识别
                reconstruction_errors.append(loss)

        # Convert the list of tensors to a single tensor
        reconstruction_errors_tensor = torch.stack(reconstruction_errors)
        return reconstruction_errors_tensor


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

# GCN-AE 实现（Encoder=GCNConv，Decoder=MLP）
class GCNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, latent_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        return x  # [N, latent_channels]


class GCNAutoEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, latent_channels=32):
        super(GCNAutoEncoder, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, latent_channels)
        self.decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)   # [N, latent]
        x_hat = self.decoder(z)                        # [N, F]
        return x_hat

class GCNAE(nn.Module):
    def __init__(
        self,
        ori_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        device,
        epochs: int = 50,
        hidden_channels: int = 64,
        latent_channels: int = 32,
        lr: float = 1e-3,
    ):
        super(GCNAE, self).__init__()
        self.device = device
        self.epochs = epochs

        self.ori_x = ori_x.to(self.device)                # [N, F]
        self.edge_index = edge_index.to(self.device)      # [2, E]
        self.edge_weight = edge_weight
        if self.edge_weight is not None:
            self.edge_weight = edge_weight.to(self.device)

        in_channels = ori_x.size(1)
        self.model = GCNAutoEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            latent_channels=latent_channels
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none')

    def fit(self):
        self.model.train()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            x_hat = self.model(self.ori_x, self.edge_index, self.edge_weight)  # [N, F]
            loss_node_feat = self.criterion(x_hat, self.ori_x)  # [N, F]
            loss = loss_node_feat.mean()
            loss.backward()
            self.optimizer.step()
            # if (epoch + 1) % 10 == 0:
            #     print(f"[GCN-AE][{epoch+1}/{self.epochs}] loss={loss.item():.4f}")

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        self.model.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        ew = edge_weight
        if ew is not None:
            ew = ew.to(self.device)

        x_hat = self.model(x, edge_index, ew)                 # [N, F]
        loss_node_feat = self.criterion(x_hat, x)             # [N, F]
        loss_node = loss_node_feat.mean(dim=1)                # [N]
        return loss_node




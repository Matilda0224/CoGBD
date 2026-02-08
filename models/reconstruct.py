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
            # 如果特征没有严格归一到 [0,1]，可以改为 Identity 或去掉激活
            nn.Sigmoid()
        )

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)   # [N, latent]
        x_hat = self.decoder(z)                        # [N, F]
        return x_hat

# GCN-AE 训练封装：GCNAE（类似你原来的 MLPAE）
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
        # 节点级 MSE，reduction='none' 保留 [N, F]
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
        """
        返回每个节点的重构误差 [N]（对 feature 维做平均）
        """
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


#### dominant #####
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv


# ===== 2) PyG-DOMINANT: Attribute + Structure 两个解码头 =====
class PyGDominant(nn.Module):
    """
    DOMINANT-style model:
      - shared GCN encoder
      - attribute decoder: MLP(z) -> x_hat
      - structure decoder: A_hat = z z^T (dense; OK for Planetoid scale)
    """
    def __init__(self, in_channels, hidden_channels=64, latent_channels=32, dropout=0.0,
                 use_sigmoid_decoder=True):
        super(PyGDominant, self).__init__()
        self.encoder = GCNEncoder(in_channels, hidden_channels, latent_channels)
        self.dropout = dropout

        last_act = nn.Sigmoid() if use_sigmoid_decoder else nn.Identity()
        self.attr_decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            last_act,
        )
        # ---- homophily decoder（新增）----
        self.homo_decoder = nn.Sequential(
            nn.Linear(latent_channels, hidden_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, in_channels),
            last_act,
        )

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)             # [N, d]
        z = F.dropout(z, p=self.dropout, training=self.training)
        x_hat = self.attr_decoder(z)                             # [N, F]
        # A_hat = z @ z.t()                                        # [N, N]
        return z,  x_hat

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_scatter import scatter_mean

def neighbor_mean(z, edge_index, num_nodes):
    """
    计算 \bar z_v = mean_{u in N(v)} z_u
    """
    src, dst = edge_index
    return scatter_mean(z[src], dst, dim=0, dim_size=num_nodes)

import torch

def edge_drop(edge_index: torch.Tensor,
              edge_weight: torch.Tensor = None,
              drop_prob: float = 0.2):
    """
    每条边以 drop_prob 概率被丢弃（不改节点数）。
    edge_index: [2, E]
    edge_weight: [E] or None
    """
    if drop_prob <= 0.0:
        return edge_index, edge_weight

    E = edge_index.size(1)
    device = edge_index.device
    keep_mask = (torch.rand(E, device=device) > drop_prob)

    edge_index_new = edge_index[:, keep_mask]
    if edge_weight is None:
        edge_weight_new = None
    else:
        edge_weight_new = edge_weight[keep_mask]
    return edge_index_new, edge_weight_new


def feature_mask_element(x: torch.Tensor, mask_prob: float = 0.2):
    """
    元素级 mask：每个 (node, dim) 以 mask_prob 概率置 0
    """
    if mask_prob <= 0.0:
        return x
    mask = (torch.rand_like(x) > mask_prob).float()
    return x * mask


def feature_mask_channel(x: torch.Tensor, mask_prob: float = 0.2):
    """
    通道级 mask：每个 feature dim 以 mask_prob 概率整体置 0（更像“删掉 trigger 维度”）
    """
    if mask_prob <= 0.0:
        return x
    F = x.size(1)
    device = x.device
    keep = (torch.rand(F, device=device) > mask_prob).float()  # [F]
    return x * keep.unsqueeze(0)


class DominantDetector(nn.Module):
    """
    - fit(): 在 poisoned graph 上无监督训练 DOMINANT（用标量 MSELoss）
    - inference(): 输出每个节点的 (score, attr_err_node, struct_err_node)
    """
    def __init__(
        self,
        args,
        ori_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        device
        # epochs: int = 50,
        # hidden_channels: int = 64,
        # latent_channels: int = 32,
        # lr: float = 5e-3,
        # a: float = 0.8,
        # dropout:float = 0.0,
        # add_self_loops_in_label: bool = True,
        # use_sigmoid_decoder: bool = True,   # 需要你在 PyGDominant 里支持这个参数
        # # ====== V1 训练扰动强度 ======
        # edge_drop_prob: float = 0.5,
        # feat_mask_prob: float = 0.5,
        # feat_mask_mode: str = "element",  # "channel" or "element"
    ):
        super(DominantDetector, self).__init__()
        self.device = device
        self.args=args
        self.a = float(self.args.a)
        self.epochs = self.args.rec_epochs

        self.ori_x = ori_x.to(self.device, dtype=torch.float32)       # [N, F]
        self.edge_index = edge_index.to(self.device, dtype=torch.long)
        self.edge_weight = edge_weight.to(self.device, dtype=torch.float32) if edge_weight is not None else None

        self.N = self.ori_x.size(0)

        # view扰动参数
        self.edge_drop_prob = float(self.args.drop_ratio)
        self.feat_mask_prob = float(self.args.mask_ratio)
        feat_mask_mode = self.args.feat_mask_mode
        assert feat_mask_mode in ["channel", "element"]
        self.feat_mask_mode = feat_mask_mode
        use_sigmoid_decoder = True
        # 训练用标量 MSE（类似 MLPAE）
        self.criterion = nn.MSELoss(reduction="mean")
        hidden = self.args.hidden
        dropout = self.args.dropout
        self.model = PyGDominant(
            in_channels=self.ori_x.size(1),
            hidden_channels=hidden,
            latent_channels=hidden,
            dropout=dropout,
            use_sigmoid_decoder=use_sigmoid_decoder,
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.ae_lr)

    def _sample_view(self):
        """
        从原 poisoned 图采样一个训练视图：
          - edge drop -> (ei_view, ew_view)
          - feature mask -> x_view
        """
        ei_view, ew_view = edge_drop(self.edge_index, self.edge_weight, drop_prob=self.edge_drop_prob)

        if self.feat_mask_mode == "channel":
            x_view = feature_mask_channel(self.ori_x, mask_prob=self.feat_mask_prob)
        else:
            x_view = feature_mask_element(self.ori_x, mask_prob=self.feat_mask_prob)

        return x_view, ei_view, ew_view

    def fit(self):
        """
        训练阶段：用标量 MSELoss（全图均值）来优化
        loss = alpha * MSE(X_hat, X) + (1-alpha) * MSE(A_hat, A)
        """
        self.model.train()
        #只在外面做一次
        # x_view, ei_view, ew_view = self._sample_view()
        x_view = self.ori_x
        ei_view = self.edge_index
        ew_view = self.edge_weight
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            # ====== 1) 采样训练视图 ======
            # x_view, ei_view, ew_view = self._sample_view()

            # z,  X_hat = self.model(self.ori_x, self.edge_index, self.edge_weight)
             # ====== 2) forward on view ======
            z, X_hat = self.model(x_view, ei_view, ew_view)

            # 标量（全图）属性 MSE
            # loss_attr = self.criterion(X_hat, self.ori_x)
             # ====== 3) attribute self-reconstruction loss (view -> view) ======
            loss_attr = self.criterion(X_hat, x_view)

            # 标量（全图）结构 MSE
            # loss_struct = self.criterion(A_hat, self.adj_label)

             # ===== 3) homophily-consistency reconstruction（新增）=====
            # z_bar = neighbor_mean(z, self.edge_index, self.N)     # [N, d]
            # X_homo_hat = self.model.homo_decoder(z_bar)            # [N, F]
            # loss_homo = self.criterion(X_homo_hat, self.ori_x)

            # ====== 4) homophily-consistency self-reconstruction loss (neighbor-mean z_bar -> x_view) ======
            z_bar = neighbor_mean(z, ei_view, self.N)               # [N, d]
            # X_homo_hat = self.model.homo_decoder(z_bar)             # [N, F]
            # loss_homo = self.criterion(X_homo_hat, x_view)

            m_x_view = neighbor_mean(x_view, ei_view, self.N)        # [N,F]
            X_homo_hat = self.model.homo_decoder(z_bar)              # [N,F]
            loss_homo = self.criterion(X_homo_hat, m_x_view)


            loss = self.a * loss_attr + (1.0 - self.a) * loss_homo
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 20 == 0 or epoch == self.epochs - 1:
                print(f"[DOMINANT][{epoch+1}/{self.epochs}] "
                      f"loss={loss.item():.4f} attr={loss_attr.item():.4f} homo={loss_homo.item():.4f}")

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """
        推理阶段：逐节点算重建误差（用于异常检测）
        返回:
          score:        [N]
          attr_err_node:[N]
          struct_err_node:[N]
        """
        self.model.eval()
        x = x.to(self.device, dtype=torch.float32)
        edge_index = edge_index.to(self.device, dtype=torch.long)
        ew = edge_weight.to(self.device, dtype=torch.float32) if edge_weight is not None else None

        # 注意：如果你 inference 用的是不同的 edge_index（比如 edge-drop 视图）
        # 那么必须重建对应的 adj_label_view；否则就用训练图的 self.adj_label
        # adj_label = self.adj_label
        # if not torch.equal(edge_index, self.edge_index):
        #     adj_label = self._build_dense_adj_label(edge_index, ew, self.N, add_self_loops=True)
        #只在外面做一次
        scores = torch.zeros(x.size(0), device=self.device, dtype=torch.float32)
        for i in range(10):
            x_view, ei_view, ew_view = self._sample_view()
            z, X_hat = self.model(x_view, ei_view, ew_view)
            attr_err_node = torch.sqrt(torch.sum((X_hat - x_view) ** 2, dim=1) + 1e-12)  # [N]
            z_bar = neighbor_mean(z, ei_view, self.N)
            # X_homo_hat = self.model.homo_decoder(z_bar)

            m_x_view = neighbor_mean(x_view, ei_view, self.N)        # [N,F]
            X_homo_hat = self.model.homo_decoder(z_bar)              # [N,F]
            homo_err_node = torch.sqrt(
                torch.sum((X_homo_hat - m_x_view) ** 2, dim=1) + 1e-12
            )
            score = self.a * attr_err_node + (1.0 - self.a) *  homo_err_node
            scores += score
        scores = scores/10

        # z, X_hat = self.model(x, edge_index, ew)

        # # ===== 节点级 attribute error =====
        # # 选你论文/实现一致的形式：L2（或 L2^2 / mean MSE）
        # # 这里用 L2：||x_v - xhat_v||_2
        # attr_err_node = torch.sqrt(torch.sum((X_hat - x) ** 2, dim=1) + 1e-12)  # [N]

        # # ===== 节点级 structure error =====
        # # ||A_{v,:} - Ahat_{v,:}||_2
        # # struct_err_node = torch.sqrt(torch.sum((A_hat - adj_label) ** 2, dim=1) + 1e-12)  # [N]
         
        # # ===== homophily-consistency error（新增）=====
        # z_bar = neighbor_mean(z, edge_index, self.N)
        # # X_homo_hat = self.model.homo_decoder(z_bar)

        # m_x_view = neighbor_mean(x, edge_index, self.N)        # [N,F]
        # X_homo_hat = self.model.homo_decoder(z_bar)              # [N,F]
        # # loss_hzomo = self.criterion(X_homo_hat, m_x_view)

        # # homo_err_node = torch.sqrt(
        # #     torch.sum((X_homo_hat - x) ** 2, dim=1) + 1e-12
        # # )
        # homo_err_node = torch.sqrt(
        #     torch.sum((X_homo_hat - m_x_view) ** 2, dim=1) + 1e-12
        # )
        # score = self.a * attr_err_node + (1.0 - self.a) *  homo_err_node
        return scores, attr_err_node,  homo_err_node

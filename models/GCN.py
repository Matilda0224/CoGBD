#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GCN(nn.Module):

    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2,device=None,layer_norm_first=False,use_ln=False,add_selfloop = True):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.add_self_loops = add_selfloop

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid, add_self_loops=self.add_self_loops))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer-2):
            self.convs.append(GCNConv(nhid,nhid,add_self_loops=self.add_self_loops))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass,add_self_loops=self.add_self_loops)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None 
        self.weight_decay = weight_decay

        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

    def forward(self, x, edge_index, edge_weight=None):
        if(self.layer_norm_first):
            x = self.lns[0](x)
        i=0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,edge_weight))
            if self.use_ln:
                x = self.lns[i+1](x)
            i+=1
            x = F.dropout(x, self.dropout, training=self.training)
        features = x
        x = self.gc2(x, edge_index,edge_weight)
        return F.log_softmax(x,dim=1), features
    
    def get_h(self, x, edge_index):
        if x.shape[1] > 1024 or x.shape[1] < 256:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
                mean = x.mean(dim=1, keepdim=True) 
                std = x.std(dim=1, keepdim=True)  
                std = torch.where(std == 0, torch.ones_like(std), std)
                x = (x - mean) / std
        else:
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
        return x
    
    # def get_h(self, x, edge_index): 
    #     for conv in self.convs:
    #             x = F.relu(conv(x, edge_index))
    #     return x

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None, gamma=None, lambda_unlearn=1.0, unlearn_mode='entropy',use_rigbd=False):
        """Train the gcn model, when idx_val is not None, pick the best model according to the validation loss.
        Parameters
        ----------
        features :
            node features
        adj :
            the adjacency matrix. The format could be torch.tensor or scipy matrix
        labels :
            node labels
        idx_train :
            node training indices
        idx_val :
            node validation indices. If not given (None), GCN training process will not adpot early stopping
        train_iters : int
            number of training epochs
        initialize : bool
            whether to initialize parameters before training
        verbose : bool
            whether to show verbose logs
        """

        self.edge_index, self.edge_weight = edge_index, edge_weight
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            if finetune==True:
                # use RIGBD
                if use_rigbd:
                    print(f'#######use RIGBD robust training#######')
                    self.rigbd_finetune(self.labels, idx_train, idx_val, attach, train_iters, verbose)
                else:
                    self.finetune(self.labels, idx_train, idx_val, attach, train_iters, verbose, gamma, lambda_unlearn=lambda_unlearn, unlearn_mode=unlearn_mode)
                # self.finetune(self.labels, idx_train, idx_val, attach, train_iters, verbose, gamma, lambda_unlearn=lambda_unlearn, unlearn_mode=unlearn_mode)
            else: 
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
        # torch.cuda.empty_cache()

    def rigbd_finetune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose):
        # idx1 = idx_train[:-len(idx_attach)]
        # idx2 = idx_train[-len(idx_attach):]
        # idx1 = [item for item in idx_train if item not in idx_attach]

        idx_train_set = set(idx_train)
        idx_attach_set = set(idx_attach)
        idx1 = list(idx_train_set - idx_attach_set)
        idx2 = idx_attach

        idx1 = torch.tensor(idx1).to(self.device)
        idx2 = torch.tensor(idx2).to(self.device)
        

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output,_ = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx1], labels[idx1]) #对 normal node正常做 nll_loss
            probs = F.softmax(output[idx2], dim=1)
            # ori
            target_probs = probs[range(len(labels[idx2])), labels[idx2]]
            loss_train = loss_train + loss_train_2
            loss_train.backward()
            optimizer.step()
        self.eval()
        self.output = output


    def finetune(self, labels, idx_train, idx_val, idx_attach, train_iters, verbose,
                gamma: torch.Tensor,
                lambda_unlearn: float = 1.0,
                unlearn_mode: str = "entropy",   # "entropy" or "maxprob"
                a_sup: float = 2.0,              # (1-gamma)^a
                b_unl: float = 2.0,              # gamma^b
                reg_nodes: torch.Tensor = None, 
                ):
        device = self.device

        # ---- to tensor ----
        if not torch.is_tensor(idx_train):
            idx_train = torch.tensor(idx_train, dtype=torch.long, device=device)
        else:
            idx_train = idx_train.to(device)
        
        gamma = gamma.to(device).clamp(0, 1)

        if idx_attach is None:
            reg_nodes = torch.arange(self.features.size(0), device=device)
        else:
            reg_nodes = idx_attach

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for it in range(train_iters):
            self.train()
            optimizer.zero_grad()

            logp, _ = self.forward(self.features, self.edge_index, self.edge_weight)  # log_softmax [N,C]

            # ===== 1) weighted supervised loss on idx_train =====
            nll = F.nll_loss(logp[idx_train], labels[idx_train], reduction='none')  # [n_train]
            w_sup = (1.0 - gamma[idx_train]).pow(a_sup).clamp_min(1e-6)            # [n_train]， 
            loss_clean = (w_sup * nll).sum() / (w_sup.sum() + 1e-12)

            # ===== 2) weighted unlearn regularizer on reg_nodes =====
            probs = torch.softmax(logp[reg_nodes], dim=1)                           # [m,C]
            w_unl = gamma[reg_nodes].pow(b_unl).clamp_min(1e-12)                    # [m]

            if unlearn_mode == "maxprob":
                reg = probs.max(dim=1).values                                       # [m]
            elif unlearn_mode == "entropy":
                reg = (probs * probs.clamp_min(1e-12).log()).sum(dim=1)             # [m]
            else:
                raise ValueError(f"Unknown unlearn_mode={unlearn_mode}")

            loss_unlearn = (w_unl * reg).sum() / (w_unl.sum() + 1e-12)

            loss = loss_clean + lambda_unlearn * loss_unlearn
            loss.backward()
            optimizer.step()

            if verbose and it % 50 == 0:
                print(f"[soft-ft] it={it} clean={loss_clean.item():.4f} unlearn={loss_unlearn.item():.4f} "
                    f"w_sup(mean)={w_sup.mean().item():.3f} w_unl(mean)={w_unl.mean().item():.3f}")

        self.eval()
        self.output = logp


    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for i in range(train_iters):
            optimizer.zero_grad()
            output,_= self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output,_ = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output
        # torch.cuda.empty_cache()

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss_val = 100
        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output,_ = self.forward(self.features, self.edge_index, self.edge_weight)
            # print("idx_train:", idx_train)
            # print("idx_train dtype:", idx_train.dtype)
            # print("idx_train shape:", idx_train.shape)
            # print("labels shape:", labels.shape)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output,_ = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])
            
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))
                print("acc_val: {:.4f}".format(acc_val))
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)
        # torch.cuda.empty_cache()


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        with torch.no_grad():
            output,_ = self.forward(features, edge_index, edge_weight)
            acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # torch.cuda.empty_cache()
        return acc_test,correct_nids

############# DOMINANT ################
import torch
from torch_scatter import scatter_mean

def neighbor_mean(feat: torch.Tensor, edge_index: torch.Tensor, num_nodes: int):
    """
    mean_{u in N(v)} feat_u, aggregated on dst nodes.
    edge_index: [2, E], where src=edge_index[0], dst=edge_index[1]
    returns: [N, D]
    """
    src, dst = edge_index
    return scatter_mean(feat[src], dst, dim=0, dim_size=num_nodes)

import torch.nn as nn
import torch.nn.functional as F

class MLPDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0,
                 use_sigmoid: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, z):
        out = self.net(z)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out

class Consistency(nn.Module):

    def __init__(self,
                 gcn_encoder,          # your GCN instance
                 nfeat: int,
                 nhid: int,
                 dec_hidden: int = 128,
                 dropout: float = 0.0,
                 use_sigmoid_x: bool = False):
        super().__init__()
        self.enc = gcn_encoder
        self.nfeat = nfeat
        self.nhid = nhid

        self.dec_x = MLPDecoder(in_dim=nhid, hidden_dim=dec_hidden, out_dim=nfeat,
                                dropout=dropout, use_sigmoid=use_sigmoid_x)

        # homo head: input is z_bar in latent space (nhid), output is m_hat in feature space (nfeat)
        self.dec_homo = MLPDecoder(in_dim=nhid, hidden_dim=dec_hidden, out_dim=nfeat,
                                   dropout=dropout, use_sigmoid=use_sigmoid_x)

    def encode(self, x, edge_index):
        # use your GCN hidden representation
        h = self.enc.get_h(x, edge_index)    # [N, nhid]
        return h

    def forward(self, x, edge_index):
        """
        returns:
          h:        [N, nhid]
          x_hat:    [N, F]  (reconstruct X)
          m_hat:    [N, F]  (reconstruct m_v = mean neighbor features)
        """
        N = x.size(0)
        h = self.encode(x, edge_index)                    # [N, nhid]
        x_hat = self.dec_x(h)                             # [N, F]

        h_bar = neighbor_mean(h, edge_index, N)           # [N, nhid]
        m_hat = self.dec_homo(h_bar)                      # [N, F]
        return h, x_hat, m_hat

from torch_geometric.utils import add_self_loops

class ConsistencyDetector(nn.Module):
    def __init__(self,
                 gcn_encoder: nn.Module,
                 a: float,
                 b: float,
                 c: float,
                 lr: float,
                 epochs: int,
                 device,
                 add_selfloop_for_mean: bool = True,
                 dec_hidden: int = 128,
                 dec_dropout: float = 0.0,
                 use_sigmoid_x: bool = False):
        super().__init__()
        self.device = device
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.epochs = int(epochs)
        self.add_selfloop_for_mean = add_selfloop_for_mean

        self.mse = nn.MSELoss(reduction="mean")

        self.gcn = gcn_encoder

        nfeat = self.gcn.nfeat
        nhid = self.gcn.hidden_sizes[0]

        self.model = Consistency(
            gcn_encoder=self.gcn,
            nfeat=nfeat,
            nhid=nhid,
            dec_hidden=dec_hidden,
            dropout=dec_dropout,
            use_sigmoid_x=use_sigmoid_x
        ).to(self.device)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

   
    def _prep_edge_index(self, edge_index, num_nodes):
        from torch_geometric.utils import add_self_loops
        edge_index = edge_index.to(self.device)
        if self.add_selfloop_for_mean:
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    def fit(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight:torch.Tensor):
        self.model.train()
        x = x.to(self.device, dtype=torch.float32)
        N = x.size(0)
        edge_index = self._prep_edge_index(edge_index, N) 

        for ep in range(self.epochs):
            self.opt.zero_grad()
            _, x_hat, m_hat = self.model(x, edge_index)

            # # m = neighbor mean of ORIGINAL x
            m = neighbor_mean(x, edge_index, N)

            loss_x = self.mse(x_hat, x)
            loss_homo = self.mse(m_hat, m)
            loss_homo_x = self.mse(m_hat, x)

            loss =  self.c * loss_x + self.a * loss_homo + self.b * loss_homo_x
            loss.backward()
            self.opt.step()

            if (ep + 1) % 20 == 0 or ep == self.epochs - 1:
                print(f"[Consistency][{ep+1}/{self.epochs}] "
                      f"loss={loss.item():.4f} node={loss_x.item():.4f} neigh={loss_homo.item():.4f} homo:{loss_homo_x.item():.4f}")

    @torch.no_grad()
    def inference(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        """
        returns:
          score: [N]
          attr_err: [N]  = ||x_hat - x||_2
          homo_err: [N]  = ||m_hat - m||_2
        """
        self.model.eval()
        x = x.to(self.device, dtype=torch.float32)
        N = x.size(0)
        edge_index = self._prep_edge_index(edge_index,  N)

        _, x_hat, m_hat = self.model(x, edge_index)
        m = neighbor_mean(x, edge_index, N)

        attr_err = torch.sqrt(torch.sum((x_hat - x) ** 2, dim=1) + 1e-12)
        neigh_err = torch.sqrt(torch.sum((m_hat - m) ** 2, dim=1) + 1e-12)
        homo_err = torch.sqrt(torch.sum((m_hat - x) ** 2, dim=1) + 1e-12)
        # score = self.a * attr_err + (1.0 - self.a) * homo_err
        score = self.c * attr_err + self.a * neigh_err + self.b * homo_err
    

        return score, attr_err, neigh_err

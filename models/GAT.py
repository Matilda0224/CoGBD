#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from copy import deepcopy
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix

class GAT(nn.Module):

    def __init__(self, nfeat, nhid, nclass, heads=8,dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True ,device=None):

        super(GAT, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GATConv(nfeat,nhid,heads,dropout=dropout)
        self.gc2 = GATConv(heads*nhid, nclass, concat=False, dropout=dropout)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None

    def forward(self, x, edge_index, edge_weight=None): 
        x = F.dropout(x, p=self.dropout, training=self.training)    # optional
        # x = F.elu(self.gc1(x, edge_index, edge_weight))   # may apply later 
        x = F.elu(self.gc1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        # x = self.gc2(x, edge_index, edge_weight)
        features = x
        x = self.gc2(x, edge_index)
      
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

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, verbose=False, finetune=False, attach=None, gamma=None, lambda_unlearn=1.0, unlearn_mode='entropy',use_rigbd=False):
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
            else: 
                self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

    # def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False):
    #     print(f'idx_train: {idx_train.shape}, {type(idx_train)}')
    #     if initialize:
    #         self.initialize()
    #     self.edge_index, self.edge_weight = edge_index, edge_weight
    #     self.features = features
    #     self.labels = torch.tensor(labels, dtype=torch.long)


    #     if idx_val is None:
    #         self._train_without_val(self.labels, idx_train, train_iters, verbose)
    #     else:
    #         self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)
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
            loss_train_2 = torch.mean(target_probs)  # Mean of probabilities of correct labels

            # 改进
            # pred_classes = probs.argmax(dim=1)
            # pred_probs = probs[range(len(pred_classes)), pred_classes]
            # loss_train_2 = torch.mean(pred_probs)


            # Combining the normal and adversarial losses
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
                # minimize sum p log p  <=> maximize entropy
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
            output,_ = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output,_ = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

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
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()



            self.eval()
            output,_ = self.forward(self.features, self.edge_index,self.edge_weight)
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


    def test(self, features, edge_index, edge_weight, labels,idx_test):
        """Evaluate GCN performance on test set.
        Parameters
        ----------
        idx_test :
            node testing indices
        """
        self.eval()
        output,_ = self.forward(features, edge_index, edge_weight)
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        # print("Test set results:",
        #       "loss= {:.4f}".format(loss_test.item()),
        #       "accuracy= {:.4f}".format(acc_test.item()))
        return float(acc_test)
    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels,idx_test):
        self.eval()
        output,_ = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test]==labels[idx_test]).nonzero().flatten()   # return a tensor
        acc_test = utils.accuracy(output[idx_test], labels[idx_test])
        return acc_test,correct_nids
# %%

#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.GCN import GCN

#%%
class GradWhere(torch.autograd.Function):
    """
    自定义前向/反向算子 ——离散化但保留梯度
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # Trigger generator: In the furture, we may use a GNN model to generate backdoor 
    # 输入: 中毒节点的特征 nfeat [m,d]
    # 输出: trigger 节点的特征 [m*nout, d], edge_weight: trigger内部可能的边（通过 GradWhere 阈值化得二值边选择）
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()

        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        
        self.layers = nn.Sequential(*layers).to(device)

        self.feat = nn.Linear(nfeat,nout*nfeat) # h -> trigger feat
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2)) # h -> trigger edge index
        self.device = device

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply #对 edge weights 做阈值化（0-1）离散化：前向：用 torch.where 把大于 thrd 的值置 1，小于 thrd 的置 0；反向：保留梯度，保证 trigger 结构仍可学习。
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        # feat [B, nout*nfeat]
        return feat, edge_weight # 这里的输出有结构

class HomoLoss(nn.Module): # 约束trigger节点的特征相似性
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args #没用到
        self.device = device
        
    def forward(self, trigger_edge_index, trigger_edge_weights, x, thrd):
        
        # trigger_edge_index：trigger和attack node的连接，以及 trigger内部的连接
        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        # 连接两边的余弦相似度
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class UGBA:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        self.trigger_index = self.get_trigger_index(args.trigger_size) # 初始化为一个 full-connected 结构
    
    def get_trigger_index(self,trigger_size): # 生成一个固定的： full-connected 子图结构模版
        edge_list = []
        edge_list.append([0,0]) # 占位，后续被替换为 （宿主节点 id, 新加 trigger 节点的起始 id)
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j,k])
        edge_index = torch.tensor(edge_list,device=self.device).long().T #edge_index
        return edge_index # [2,E] 第一行为 起点节点编号；第二行为 终点节点编号

    def get_trojan_edge(self,start, idx_attach, trigger_size): # 将trigger挂载到 attack节点 上，使trigger的节点和边的编号和原图统一
        edge_list = []
        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0,0] = idx 
            edges[1,0] = start
            edges[:,1:] = edges[:,1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list,dim=1)
        # to undirected
        # row, col = edge_index
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1],edge_index[0]])
        edge_index = torch.stack([row,col])

        return edge_index
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval() # 利用已经训练好的 trigger generator

        # 1. 生成trigger node feature 和 edge weights
        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd) # may revise the process of generate
        
        # 2. 拼接 trigger 的边权向量 (和attack node 之间的边权重为 1)
        trojan_weights = torch.cat([torch.ones([len(idx_attach),1],dtype=torch.float,device=device),trojan_weights],dim=1)
        trojan_weights = trojan_weights.flatten()

        # 3. 整理 trigger 节点特征矩阵：把所有 trigger 节点的特征展平堆叠为一个矩阵 [num_trigger_nodes, d]
        trojan_feat = trojan_feat.view([-1,features.shape[1]])

        # 4. 构建 trigger 边索引矩阵: 每个 attach 节点都会被连接一个 trigger 子图,trigger 子图编号从 len(features) 开始递增
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,self.args.trigger_size).to(device)

        # 5. 拼接回原图，生成“中毒图”
        update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
        update_feat = torch.cat([features,trojan_feat])
        update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):
        # trigger generator 与 shadow model 交替优化参数
        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index
        self.edge_weights = edge_weight
        
        # 1. initial a shadow model
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        # 2. initalize a trojanNet to generate trigger
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.shadow_lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.trojan_lr, weight_decay=args.weight_decay)

    
        # 3. change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # 4. 初始化‘trigger’结构模版： 为每个 attach 节点构造一个 trigger 子图（固定结构 + 偏移索引），并连上宿主节点。
        # 注意：这里trigger节点的具体feature和边权还没生成
        # get the trojan edges, which include the target-trigger edge and the edges among trigger
        trojan_edge = self.get_trojan_edge(len(features),idx_attach,args.trigger_size).to(self.device)
        # update the poisoned graph's edge index
        poison_edge_index = torch.cat([edge_index,trojan_edge],dim=1)


        # furture change it to bilevel optimization
        loss_best = 1e8
        for i in range(args.trojan_epochs):
            self.trojan.train()# 为什么这里是 trojan.train()?
            for j in range(self.args.inner):
                # 5. 内层： 更新 shadow_model
                optimizer_shadow.zero_grad()
                # 5.1： 得到trigger generator生成的trigger特征和边权重 （ args.thrd: 二值化阈值 >thrd → 边存在)
                trojan_feat, trojan_weights = self.trojan(features[idx_attach],args.thrd) # may revise the process of generate
                # 为每个宿主节点与第一个 trigger 节点间的边权设为 1（保证触发器连接）; 然后展开为 1D 向量，与 trigger 边索引顺序匹配。
                trojan_weights = torch.cat([torch.ones([len(trojan_feat),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
                trojan_weights = trojan_weights.flatten()
                # 拼接新的节点特征, 拼接边权 (重复两次是因为图被双向化)
                trojan_feat = trojan_feat.view([-1,features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights]).detach() #虽然self.trojan.train()，但是并不会对trojan进行反向传播
                poison_x = torch.cat([features,trojan_feat]).detach()
                # shadow_model返回 log_softmax之后的结果
                output,_ = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                # 内层目标： Shadow 模型在带 trigger 的毒化图上训练， 能正确预测带有trigger的点为target class
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                loss_inner.backward()
                optimizer_shadow.step()

            # 检查shadow_model的性能
            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            
            # 6. 外层： 优化 trigger generator
            # involve unlabeled nodes in outter optimization
            self.trojan.eval() # 为什么这里却是 eval()? 这里根本不需要把（SPEA中去掉了）
            optimizer_trigger.zero_grad()

            rs = np.random.RandomState(self.args.seed)
            # 6.1. SPEAR follow it: 从unlabel idx中选择512个节点一起参与 trigger generator 更新
            idx_outter = torch.cat([idx_attach,idx_unlabeled[rs.choice(len(idx_unlabeled),size=args.outter_size,replace=False)]])
            # 6.2. 重新生成trigger 
            trojan_feat, trojan_weights = self.trojan(features[idx_outter],self.args.thrd) # may revise the process of generate

            trojan_weights = torch.cat([torch.ones([len(idx_outter),1],dtype=torch.float,device=self.device),trojan_weights],dim=1)
            trojan_weights = trojan_weights.flatten()

            trojan_feat = trojan_feat.view([-1,features.shape[1]])
            # 6.3. 将新的 trigger 节点、边、权拼接到图上
            trojan_edge = self.get_trojan_edge(len(features),idx_outter,self.args.trigger_size).to(self.device)

            update_edge_weights = torch.cat([edge_weight,trojan_weights,trojan_weights])
            update_feat = torch.cat([features,trojan_feat])
            update_edge_index = torch.cat([edge_index,trojan_edge],dim=1)
           
            output,_ = self.shadow_model(update_feat, update_edge_index, update_edge_weights)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class
            # 6.4. 让 shadow model 把中毒节点 + 未标注节点尽可能预测为目标类
            loss_target = self.args.target_loss_weight *F.nll_loss(output[torch.cat([idx_train,idx_outter])],
                                    labels_outter[torch.cat([idx_train,idx_outter])])
            loss_homo = 0.0

            if(self.args.homo_loss_weight > 0):
                # 6.5. 加入同质性loss: 让 trigger 节点间的特征更相似、更自然
                loss_homo = self.homo_loss(trojan_edge[:,:int(trojan_edge.shape[1]/2)],\
                                            trojan_weights,\
                                            update_feat,\
                                            self.args.homo_boost_thrd)
            
            loss_outter = loss_target + self.args.homo_loss_weight * loss_homo

            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if args.debug and i % 50 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, homo loss: {:.5f} '\
                        .format(i, loss_inner, loss_target, loss_homo))
                print("acc_train_clean: {:.4f}, ASR_train_attach: {:.4f}, ASR_train_outter: {:.4f}"\
                        .format(acc_train_clean,acc_train_attach,acc_train_outter))
        if args.debug:
            print("load best weight based on the loss outter")
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

        # torch.cuda.empty_cache()

    def get_poisoned(self):

        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:,poison_edge_weights>0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights>0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels

# %%

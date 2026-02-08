#%%
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from models.GCN import GCN
import numpy as np
import os

class GraphTrojanNet(nn.Module):
    # 一个小型MLP，用于生成trigger的feature
    # 输入 node embedding (中毒节点的 node embedding), 输出 trigger embedding
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, dim_num, layernum=2, dropout=0.00):
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

        self.feat = nn.Linear(nfeat,dim_num)
        self.edge = nn.Linear(nfeat, int(dim_num*(dim_num-1)/2))
        self.device = device

    def forward(self, input, thrd):  #这个thrd没有用到？
        self.layers = self.layers
        h = self.layers(input)
        feat = self.feat(h)
        return feat

class HomoLoss(nn.Module): #约束 injected trigger 内部一致性（这是UGBA中的约束，这里没有用到）
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np
class SPEAR:

    def __init__(self,args, device):
        self.args = args
        self.device = device
        self.weights = None
        
    def inject_trigger(self, idx_attach, features,edge_index,edge_weight,device):
    # 输入：idx_attach 中毒节点index 和原图结构 features, edge_index
    # 输出L修改后的特征节点 update_feat,和unchanged 的 edge_index（因为结构其实没有动）
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval() #已经训练好了，可以直接使用
        self.shadow_model.eval()
        embed = self.shadow_model.get_h(features, edge_index) # 先到的node emb
        # 这个thrd没有用到
        trojan_feat = self.trojan(embed[idx_attach],self.args.thrd) # may revise the process of generate
        update_edge_weights = edge_weight.clone()
        update_feat = features.clone()

        args = self.args
        dim_num = args.alpha_int
        if args.dataset == 'Cora':
            sage_epoch = '32'
        elif args.dataset == 'Pubmed':
            sage_epoch = '128'
        elif args.dataset == 'ogbn-arxiv':
            sage_epoch = '2048'
        elif args.dataset == 'Flickr':
            sage_epoch = '64'
        elif args.dataset == 'Computers':
            sage_epoch = '128'
        elif args.dataset == 'Photo':
            sage_epoch = '128'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f'../save_selected_feature/{args.dataset}/val_{sage_epoch}.npy')
        dim_all = np.load(file_path)
        dim = np.argsort(dim_all)[::-1][:dim_num]
        dim = dim.copy()
        update_feat[idx_attach[:, None], dim] = trojan_feat.detach() # 将对应维度 dim 替换为 trigger的feature

        update_edge_index = edge_index.clone()

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights # edge_index前后不变

    def get_h(self, x, edge_index): # SPEAR的实现中没有用到：get_h在GCN内部，共享aggregation参数和结果效果比普通 aggregation好

        for conv in self.shadow_model.convs:
            x = F.relu(conv(x, edge_index))
        return x


    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach,idx_unlabeled):

        args = self.args
        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]],device=self.device,dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features 
        self.edge_index = edge_index
        self.edge_weights = edge_weight #都是训练图，不是完整图
        
        # 1. initial a shadow model (模拟 victim model)：对应 L_inner: shadow 模型在 poisoned data 上的训练损失
        self.shadow_model = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden,
                         nclass=labels.max().item() + 1,
                         dropout=0.0, device=self.device).to(self.device)
        
        # 2. feature attack in alpha_int dimension
        dim_num = args.alpha_int
        if args.dataset in ['Cora','Citeseer']:
            sage_epoch = '32'
        elif args.dataset in ['Pubmed', 'Physics']:
            sage_epoch = '128'
        elif args.dataset == 'ogbn-arxiv':
            sage_epoch = '2048'
        elif args.dataset in ['Flickr', 'Computers', 'Photo']:
            sage_epoch = '64'
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f'../save_selected_feature/{args.dataset}/val_{sage_epoch}.npy')
        dim_all = np.load(file_path)
        dim = np.argsort(dim_all)[::-1][:dim_num]
        dim = dim.copy()

        embed = self.shadow_model.get_h(features, edge_index)
        # 3.初始化 trigger 生成器：对应 L_outer: 攻击成功率 + 隐蔽性
        self.trojan = GraphTrojanNet(self.device, embed.shape[1], dim_num, layernum=2).to(self.device)
        self.homo_loss = HomoLoss(self.args,self.device)

        optimizer_shadow = optim.Adam(self.shadow_model.parameters(), lr=args.shadow_lr, weight_decay=args.weight_decay)
        optimizer_trigger = optim.Adam(self.trojan.parameters(), lr=args.trojan_lr, weight_decay=args.weight_decay)

    
        # change the labels of the poisoned node to the target class
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # initialization
        poison_x = features.clone().detach()
        loss_best = 1e8

        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):
                #4. 内层: 优化shadow，让 shadow 模型在当前的 污染训练图 上（包含 attach 节点）尽量拟合正确标签
                optimizer_shadow.zero_grad()

                #trojan_feat is a alpha_int dimension feature that serves as trigger and will be implemented in fearture space
                embed = self.shadow_model.get_h(poison_x, edge_index) # 通过shadow model得到节点嵌入
                trojan_feat= self.trojan(embed[idx_attach],args.thrd) # Trojan 生成triggerr征 （
                poison_edge_weights = edge_weight.detach()
                poison_x = features.clone().detach() #这里为什么？
                poison_x[idx_attach[:, None], dim] = trojan_feat.detach() # 将对应维度 替换中毒node emb，得到污染特征
                
                output,_ = self.shadow_model(poison_x, edge_index, poison_edge_weights) # 
                
                # Shadow model 适应被污染的数据分布
                loss_inner = F.nll_loss(output[torch.cat([idx_train,idx_attach])], self.labels[torch.cat([idx_train,idx_attach])]) # add our adaptive loss
                
                loss_inner.backward()
                optimizer_shadow.step()

            
            acc_train_clean = utils.accuracy(output[idx_train], self.labels[idx_train])
            acc_train_attach = utils.accuracy(output[idx_attach], self.labels[idx_attach])
            
            # 5. 外层： 优化trigger生成器，在保证隐蔽性的同时，让攻击节点（attach）和随机挑的 unlabeled 节点（outter）被模型误导到目标类别
            optimizer_trigger.zero_grad()
            rs = np.random.RandomState(self.args.seed)

            # 随机挑选 部分 未标记的节点
            outter = idx_unlabeled[rs.choice(len(idx_unlabeled),size=args.outter_size,replace=False)]

            # 组合中毒节点 + 随机挑选的未标记节点
            idx_outter = torch.cat([idx_attach,outter])

            embed = self.shadow_model.get_h(poison_x, edge_index) # 通过shadow model得到节点嵌入
            # Trojan 为中毒节点+随机节点生成 trigger特征
            trojan_feat = self.trojan(embed[idx_outter],args.thrd) # may revise the process of generate
        
            update_feat = features.clone()
            update_feat[idx_outter[:, None], dim] = trojan_feat # 替换中毒+随机node emb，得到更新后的特征

            output,_ = self.shadow_model(update_feat, edge_index, edge_weight)

            labels_outter = labels.clone()
            labels_outter[idx_outter] = args.target_class #随机挑选的节点也要替换class

            loss_sim = 1 - F.cosine_similarity(update_feat, self.features).mean() # 约束生成的特征与原特征尽量接近

            loss_target = self.args.target_loss_weight *F.nll_loss(output[torch.cat([idx_train,idx_outter])],
                                    labels_outter[torch.cat([idx_train,idx_outter])]) # 约束生成的特征要尽量保持预测准确

            # loss_target：攻击有效性；self.args.homo_loss_weight：攻击隐蔽性
            loss_outter = loss_target  + self.args.homo_loss_weight * loss_sim

            loss_outter.backward()
            optimizer_trigger.step()
            acc_train_outter =(output[idx_outter].argmax(dim=1)==args.target_class).float().mean()

            if loss_outter<loss_best:
                self.weights = deepcopy(self.trojan.state_dict())
                loss_best = float(loss_outter)

            if args.debug and i % 50 == 0:
                print('Epoch {}, loss_inner: {:.5f}, loss_target: {:.5f}, loss_sim: {:.5f}'.format(i, loss_inner, loss_target, loss_sim))
                print("ACC: {:.4f}, ASR_train: {:.4f}".format(acc_train_clean,acc_train_attach,acc_train_outter))
                
        if args.debug:
            print(f"load best weight based on the loss outter{loss_best}")
        print(f'dim: {dim}')
        self.trojan.load_state_dict(self.weights)
        self.trojan.eval()

        # os.makedirs("checkpoints", exist_ok=True)
        # PATH = f"checkpoints/model_weights_{args.dataset}.pth"

        # # 假设 self.trojan 是训练好的 GraphTrojanNet/torch.nn.Module
        # torch.save(self.trojan.state_dict(), PATH)
        # print("Saved trojan state_dict to", PATH)


    def get_poisoned(self):
        with torch.no_grad():
            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach,self.features,self.edge_index,self.edge_weights,self.device)
        poison_labels = self.labels #只对中毒节点的label进行修改
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels


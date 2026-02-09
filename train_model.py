#%%
import time
import argparse
import numpy as np
import torch
from models.GCN import GCN
import torch_geometric
from torch_geometric.datasets import Planetoid,Reddit2,Flickr,PPI
#%%

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--dataset', type=str, default='Pubmed', 
                    help='Dataset',
                    choices=['Cora','Citeseer','Pubmed','PPI','Flickr','ogbn-arxiv','Reddit','Reddit2','Yelp','Computers', 'Photo'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--test_model', type=str, default='guard',
                    choices=['GCN','GAT','GraphSage','guard'],
                    help='Model used to attack')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train.')
parser.add_argument("--layer", type=int, default=2)
# parameters of mvae
parser.add_argument('--rec_epochs', type=int,  default=100,
                    help='Number of epochs to train auto encoder.')
parser.add_argument('--ae_thr', type=int,  default=97,
                    help='od thre')
parser.add_argument('--c', type=float,  default=1,
                    help='weight of attri score')                
parser.add_argument('--a', type=float,  default=0.5,
                    help='weight of homo score.')
parser.add_argument('--b', type=float,  default=1,
                    help='weight of homo-attribute score.')
parser.add_argument('--drop_ratio', type=float,  default=0.1)
parser.add_argument('--mask_ratio', type=float,  default=0.1)
parser.add_argument('--feat_mask_mode',type=str, default='channel', choices=['channel', 'element'])
parser.add_argument('--conf_tau',  type=float,  default=1.0, help='控制node置信度/gamma的陡峭程度')
parser.add_argument('--lambda_unlearn',  type=float,  default=1.0)
parser.add_argument('--unlearn_mode',  type=str,  default='entropy', choices=['entropy', 'maxprob'])
parser.add_argument('--use_rigbd_training', action='store_true', default=False)
parser.add_argument('--graph_encoder', type=str, default='GCN', choices=['GCN','GAT','GraphSage','GNNGuard','RobustGCN'], help='Model used in encoder of detector.')
parser.add_argument('--ae_lr', type=float, default=0.01, help='Initial learning rate.')


args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if args.cuda else "cpu")
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args)

#%%
from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Flickr
from torch_geometric.transforms import NormalizeFeatures
transform = T.Compose([T.NormalizeFeatures()])

if args.dataset in ['Cora', 'Pubmed', 'Citeseer']:
    dataset = Planetoid(root='./data/', \
                        name=args.dataset,\
                        transform=transform)
elif args.dataset == 'ogbn-arxiv':
    torch.serialization.add_safe_globals([
        torch_geometric.data.data.Data,
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
    ])
    from ogb.nodeproppred import PygNodePropPredDataset

    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv', root='./data/')
    split_idx = dataset.get_idx_split() 
elif args.dataset == 'Flickr':
    dataset = Flickr(root='./data/Flickr', transform=transform)
elif args.dataset in ['Computers', 'Photo']:
    dataset = Amazon(root='./data/Amazon', name=args.dataset, transform=transform)
elif args.dataset == 'Physics':
    dataset = Coauthor(root='./data/Coauthor', name='Physics', transform=transform)

data = dataset[0].to(device)

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    # dataset[0].train_mask = torch.zeros(nEdge, dtype=torch.bool).to(device)
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)
# we build our own train test split 
#%% 
from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
#%%
import os
from models.construct import model_construct

from help_funcs import detect_and_prune_by_homodominant, detect_by_detector
train_edge_index,_ , abnormal_nodes, gamma, detector = detect_and_prune_by_homodominant(args, data, data.x, train_edge_index, None,device)
# poison_edge_index, poison_edge_weights, abnormal_nodes = detect_by_dominant_and_prune(args,poison_edge_index,poison_edge_weights,poison_x,device)
# _, _, abnormal_nodes = reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,device)
print(f'after DOMINANT+ prune:rain_edge_index: {train_edge_index.shape}')
print(f'abnormal_nodes: {abnormal_nodes}')
gamma_attach = gamma[abnormal_nodes]  # [len(idx2)
print(
    "gamma_attach stats:",
    "min", float(gamma_attach.min()),
    "max", float(gamma_attach.max()),
    "mean", float(gamma_attach.mean()),
    "std", float(gamma_attach.std()),
)

rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=5)
list_ca = []
avg_time = 0
abnormal_nodes = abnormal_nodes[abnormal_nodes <  data.y.shape[0]]
for seed in seeds: 
    args.seed = seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benign_model = model_construct(args,args.test_model,data,device).to(device) 
    t_total = time.time()
    print(f"on seed :{seed}:")
    print("Length of training set: {}".format(len(idx_train)))
    # benign_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=False)
    benign_model.fit(data.x, train_edge_index, None, data.y, idx_train, idx_val,train_iters=args.epochs,verbose=True, finetune=True, attach=abnormal_nodes, gamma = gamma, lambda_unlearn = args.lambda_unlearn, unlearn_mode=args.unlearn_mode, use_rigbd = args.use_rigbd_training)
    print("Training benign model Finished!")
    t_total = time.time() - t_total
    print("Total time elapsed: {:.4f}s".format(t_total))

    benign_ca = benign_model.test(data.x, data.edge_index, None, data.y,idx_clean_test)
    print("Benign CA: {:.4f}".format(benign_ca))
    
    list_ca.append(benign_ca)
    avg_time += t_total
    
    benign_model = benign_model.cpu()

ca_arr = np.array([t.item() if torch.is_tensor(t) else t for t in list_ca], dtype=float)

n = len(list_ca)  

mean_ca = np.mean(ca_arr)
avg_time = avg_time / n

sem_ca = np.std(ca_arr, ddof=1) / np.sqrt(n)

print("======== Overall Results ========")
print(f"CA: {mean_ca:.4f} ± {sem_ca:.4f}")    
print("Avg time elapsed: {:.4f}s".format(avg_time))
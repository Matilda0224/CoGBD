import argparse
import numpy as np
import torch

import torch_geometric
from torch_geometric.datasets import Planetoid,Flickr,Amazon
# from ogb.nodeproppred import PygNodePropPredDataset
from help_funcs import reconstruct_prune_unrelated_edge


# from torch_geometric.loader import DataLoader
from help_funcs import prune_unrelated_edge
# from select_sample import select

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=True, help='debug mode')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('-num_seeds', type=int, default=5, help='Random seed.')
parser.add_argument('--model', type=str, default='GCN', help='model',
                    choices=['GCN','GAT','GraphSage'])
parser.add_argument('--dataset', type=str, default='Cora', 
                    help='Dataset',
                    choices=['Cora','Pubmed','Flickr','ogbn-arxiv'])
parser.add_argument('--train_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--thrd', type=float, default=0.5)
parser.add_argument('--target_class', type=int, default=0)
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int,  default=200, help='Number of epochs to train benign and backdoor model.')
parser.add_argument('--trojan_epochs', type=int,  default=400, help='Number of epochs to train trigger generator.')
parser.add_argument('--inner', type=int,  default=1, help='Number of inner')

parser.add_argument('--trigger', type=str, default='UGBA',
                    choices=['SPEAR', 'UGBA', 'DPGBA', 'GTA'])
parser.add_argument('--ae_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--detector_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--shadow_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--trojan_lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--use_vs_number', action='store_true', default=True,
                    help="if use detailed number to decide Vs")
parser.add_argument('--vs_ratio', type=float, default=0,
                    help="ratio of poisoning nodes relative to the full graph")
parser.add_argument('--vs_number', type=int, default=40,
                    help="number of poisoning nodes relative to the full graph")
parser.add_argument('--defense_mode', type=str, default="dominant",
                    choices=['prune', 'none','reconstruct', 'dominant'],
                    help="Mode of defense")
parser.add_argument('--prune_thr', type=float, default=0.8,
                    help="Threshold of prunning edges")
parser.add_argument('--target_loss_weight', type=float, default=1,
                    help="Weight of optimize outter trigger generator")
parser.add_argument('--dis_weight', type=float, default=1,
                    help="Weight of cluster distance")
parser.add_argument('--test_model', type=str, default='GCN',
                    choices=['GCN','GAT','GraphSage','GNNGuard','RobustGCN'],
                    help='Model used to attack')
parser.add_argument('--device_id', type=int, default=0,
                    help="Threshold of prunning edges")

# setting of UGBA, DPGBA, GTA
parser.add_argument('--trigger_size', type=int, default=3,
                    help="Number of feature dimensions to perturb")
# setting of UGBA
parser.add_argument('--homo_boost_thrd', type=float, default=0.8,
                    help="T: Threshold of increase similarity")
parser.add_argument('--homo_loss_weight', type=float, default=0.1,
                    help="𝛽:{0,50,100,150,200} for UGBA; 0.1 for SPEAR")
# setting of DPGBA
parser.add_argument('--range', type=float, default=1.0,
                    help="ratio of poisoning nodes relative to the full graph")     
parser.add_argument('--k', type=int, default=20, help='inner steps. fixed 20 for all')
parser.add_argument('--weight_target', type=float, default=1,
                    help="Weight of attack loss")
parser.add_argument('--weight_ood', type=float, default=1,
                    help="Weight of ood constraint")
parser.add_argument('--weight_targetclass', type=float, default=1,
                    help="Weight of enhancing attack loss")             
# setting of SPEAR
parser.add_argument('--alpha', type=float, default=0.02,
                    help="Ratio of feature dimensions to perturb")
parser.add_argument('--alpha_int', type=int, default=30,
                    help="Number of feature dimensions to perturb")
parser.add_argument('--outter_size', type=int, default=512,
                    help="Number of outter samples")

# setting of CoGBD
parser.add_argument('--rec_epochs', type=int,  default=100,
                    help='Number of epochs to train auto encoder.')
parser.add_argument('--ae_thr', type=int,  default=97,
                    help='od thre')
parser.add_argument('--c', type=float,  default=1,
                    help='weight of attri score')                
parser.add_argument('--a', type=float,  default=0.5,
                    help='weight of neighborhood score.')
parser.add_argument('--b', type=float,  default=1,
                    help='weight of homophily score.')
parser.add_argument('--conf_tau',  type=float,  default=1.0 )
parser.add_argument('--lambda_unlearn',  type=float,  default=1.0)
parser.add_argument('--unlearn_mode',  type=str,  default='entropy', choices=['entropy', 'maxprob'])
parser.add_argument('--use_rigbd_training', action='store_true', default=False)
parser.add_argument('--graph_encoder', type=str, default='GCN', choices=['GCN','GAT','GraphSage','GNNGuard','RobustGCN'], help='Model used in encoder of detector.')


args = parser.parse_known_args()[0]
args.cuda =  not args.no_cuda and torch.cuda.is_available()
device = torch.device(('cuda:{}' if torch.cuda.is_available() else 'cpu').format(args.device_id))

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
print(args)

from torch_geometric.utils import to_undirected
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, Flickr
from torch_geometric.transforms import NormalizeFeatures
transform = T.Compose([T.NormalizeFeatures()])

if args.dataset in ['Cora', 'Pubmed']:
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

data = dataset[0].to(device)
print(f'data statistics: data.x:  {data.x.shape}, data.edge_index:{data.edge_index.shape}')

if(args.dataset == 'ogbn-arxiv'):
    nNode = data.x.shape[0]
    setattr(data,'train_mask',torch.zeros(nNode, dtype=torch.bool).to(device))
    data.val_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.test_mask = torch.zeros(nNode, dtype=torch.bool).to(device)
    data.y = data.y.squeeze(1)

from utils import get_split
data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,data,device)

from torch_geometric.utils import to_undirected
from utils import subgraph
data.edge_index = to_undirected(data.edge_index)
train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]


from models.construct import model_construct

unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
if(args.use_vs_number):
    size = args.vs_number
else:
    size = int((len(data.test_mask)-data.test_mask.sum())*args.vs_ratio)
print("Attach Nodes:{}".format(size))
assert size>0, 'The number of selected trigger nodes must be larger than 0!'
# here is randomly select poison nodes from unlabeled nodes
# idx_attach = select(data,args,idx_train,idx_val,device).to(device)
if args.trigger == 'SPEAR':
    from select_sample import spear_select
    idx_attach = spear_select(data,args,idx_train,idx_val,device).to(device)
elif args.trigger == 'UGBA':
    from select_sample import cluster_degree_selection_seperate_fixed, cluster_degree_selection
    idx_attach = cluster_degree_selection(args,data,idx_train,idx_val,idx_clean_test,unlabeled_idx,train_edge_index,size,device)
    # idx_attach = torch.LongTensor(idx_attach).to(device)
elif args.trigger in ['GTA', 'DPGBA']:
    from select_sample import obtain_attach_nodes
    idx_attach = obtain_attach_nodes(args,unlabeled_idx,size)
print("idx_attach: {}".format(idx_attach))

unlabeled_idx = torch.tensor(list(set(unlabeled_idx.cpu().numpy()) - set(idx_attach.cpu().numpy()))).to(device)
# print(unlabeled_idx)

# model = Backdoor(args,device)
if args.trigger == 'SPEAR':
    from attackers.SPEAR import SPEAR
    model = SPEAR(args,device)
elif args.trigger == 'UGBA':
    from attackers.UGBA import UGBA
    model = UGBA(args,device)
elif args.trigger == 'DPGBA':
    from attackers.DPGBA import DPGBA
    model = DPGBA(args,device)
elif args.trigger == 'GTA':
    from attackers.GTA import GTA
    model = GTA(args,device)
model.fit(data.x, train_edge_index, None, data.y, idx_train,idx_attach, unlabeled_idx)
poison_x, poison_edge_index, poison_edge_weights, poison_labels = model.get_poisoned()
print(f'****** poison_x:{poison_x.shape}, poison_edge_index:{poison_edge_index.shape}') # x.shape [2708, 1433]  [2, 6898]


import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli

mask = data.y[idx_attach] != args.target_class
mask = mask.to(device)
print('Number of poisoned target nodes', mask.sum())
## only attack those has groud truth labels != target_class ##
idx_attach = idx_attach[(data.y[idx_attach] != args.target_class).nonzero().flatten()]
bkd_tn_nodes = torch.cat([idx_train,idx_attach]).to(device) 
known_nodes = torch.cat([idx_train,idx_attach]).to(device)
predictions = []
# edge weight for clean edge_index, may use later #
edge_weight = torch.ones([data.edge_index.shape[1]],device=device,dtype=torch.float)

#### train a backdoored model on poisoned graph #### 
test_model = model_construct(args,args.test_model,data,device).to(device) 
test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs, verbose=False)
test_model.eval()

clean_acc = test_model.test(poison_x,poison_edge_index, poison_edge_weights,poison_labels,idx_attach)
output_clean,_ = test_model(poison_x,poison_edge_index,poison_edge_weights)
ori_predict = torch.exp(output_clean[known_nodes])

induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
ca = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()
output,_ = test_model(induct_x,induct_edge_index,induct_edge_weights)

print("****Before Defense****")
train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
print(f'ASR:{train_attach_rate}', f'Flip-ASR:{flip_asr}')
print("flip_asr: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
asr = train_attach_rate
print("CA: {:.4f}".format(ca))

from help_funcs import detect_and_prune_by_consistency_detector, detect_by_detector
poison_edge_index, poison_edge_weights, abnormal_nodes, gamma, detector = detect_and_prune_by_consistency_detector(args, data, poison_x, poison_edge_index, poison_edge_weights,device)


true_attach = set(idx_attach.cpu().numpy().tolist())
true_trigger = set(torch.arange(data.num_nodes, poison_x.shape[0]).cpu().numpy().tolist())
true_poisoned = true_poisoned = true_attach | true_trigger 

predicted_poisoned = set(abnormal_nodes.cpu().numpy().tolist())

def compute_metrics(true_set, pred_set):
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    return tp, fp, fn, recall, precision

# ===  attach nodes ===
tp_a, fp_a, fn_a, recall_a, precision_a = compute_metrics(true_attach, predicted_poisoned)
# ===  trigger node ===
tp_t, fp_t, fn_t, recall_t, precision_t = compute_metrics(true_trigger, predicted_poisoned)
# ===  all backdoor-affected nodes ===
tp_all, fp_all, fn_all, recall_all, precision_all = compute_metrics(true_poisoned, predicted_poisoned)
print(f"number of true attached nodes: {len(true_attach)}")
print(f"number of true trigger nodes: {len(true_trigger)}")
print(f"number of true poisoned nodes: {len(true_poisoned)}")
print(f"number of detected nodes: {len(predicted_poisoned)}\n")

list_asr = [] 
list_ca = []
rs = np.random.RandomState(args.seed)
seeds = rs.randint(1000,size=5)

abnormal_nodes = abnormal_nodes[abnormal_nodes <  poison_labels.shape[0]]
print(f'for robusting training, num of abnormal nodes: {len(abnormal_nodes)}')
for seed in seeds:
    args.seed = seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    test_model = model_construct(args,args.test_model,data,device).to(device) 
    test_model.fit(poison_x,poison_edge_index, poison_edge_weights, poison_labels, bkd_tn_nodes, idx_val,train_iters=args.epochs,verbose=True, finetune=True, attach=abnormal_nodes, gamma = gamma, lambda_unlearn = args.lambda_unlearn, unlearn_mode=args.unlearn_mode, use_rigbd = args.use_rigbd_training)
   
    induct_edge_index = torch.cat([poison_edge_index,mask_edge_index],dim=1)
    induct_edge_weights = torch.cat([poison_edge_weights,torch.ones([mask_edge_index.shape[1]],dtype=torch.float,device=device)])
    
    ca = test_model.test(poison_x,induct_edge_index,induct_edge_weights,data.y,idx_clean_test)

    induct_x, induct_edge_index,induct_edge_weights = model.inject_trigger(idx_atk,poison_x,induct_edge_index,induct_edge_weights,device)
    induct_x, induct_edge_index,induct_edge_weights = induct_x.clone().detach(), induct_edge_index.clone().detach(),induct_edge_weights.clone().detach()

    induct_edge_index,induct_edge_weights = detect_by_detector(args, detector, induct_x, induct_edge_index,induct_edge_weights)

    output,_ = test_model(induct_x,induct_edge_index,induct_edge_weights)
    
    train_attach_rate = (output.argmax(dim=1)[idx_atk]==args.target_class).float().mean()
    flip_idx_atk = idx_atk[(data.y[idx_atk] != args.target_class).nonzero().flatten()]
    flip_asr = (output.argmax(dim=1)[flip_idx_atk]==args.target_class).float().mean()
   
    print(f"****After Defense, on Seed{seed}****")
    print(f'ASR:{train_attach_rate}', f'Flip-ASR:{flip_asr}')
    print("flip_asr: {:.4f}/{} nodes".format(flip_asr,flip_idx_atk.shape[0]))
    print("CA: {:.4f}".format(ca))
    list_asr.append(train_attach_rate)
    list_ca.append(ca)
    

asr_arr = np.array([t.item() if torch.is_tensor(t) else t for t in list_asr], dtype=float)
ca_arr = np.array([t.item() if torch.is_tensor(t) else t for t in list_ca], dtype=float)
n = len(list_asr) 

mean_asr = np.mean(asr_arr)
mean_ca = np.mean(ca_arr)

sem_asr = np.std(asr_arr, ddof=1) / np.sqrt(n)
sem_ca = np.std(ca_arr, ddof=1) / np.sqrt(n)
print("======== Overall Results ========")
print(f"ASR: {mean_asr:.4f} ± {sem_asr:.4f}")
print(f"CA: {mean_ca:.4f} ± {sem_ca:.4f}")
print("=== Detection Performance ===")
print(f"[Attach Nodes]  TP: {tp_a}, FP: {fp_a}, FN: {fn_a}, Recall: {recall_a:.4f}, Precision: {precision_a:.4f}")
print(f"[Trigger Nodes] TP: {tp_t}, FP: {fp_t}, FN: {fn_t}, Recall: {recall_t:.4f}, Precision: {precision_t:.4f}")
print(f"[All Poisoned]  TP: {tp_all}, FP: {fp_all}, FN: {fn_all}, Recall: {recall_all:.4f}, Precision: {precision_all:.4f}")
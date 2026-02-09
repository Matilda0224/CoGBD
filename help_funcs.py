import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj,dense_to_sparse
import torch
import scipy.sparse as sp
from models.reconstruct import MLPAE
from models.DOMINANT import Dominant

def prune_unrelated_edge(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    # update structure
    updated_edge_index = edge_index[:,edge_sims>args.prune_thr]
    updated_edge_weights = edge_weights[edge_sims>args.prune_thr]
    return updated_edge_index,updated_edge_weights

def prune_unrelated_edge_isolated(args,edge_index,edge_weights,x,device,large_graph=True):
    edge_index = edge_index[:,edge_weights>0.0].to(device)
    edge_weights = edge_weights[edge_weights>0.0].to(device)
    x = x.to(device)
    # calculate edge simlarity
    if(large_graph):
        edge_sims = torch.tensor([],dtype=float).cpu()
        N = edge_index.shape[1]
        num_split = 100
        N_split = int(N/num_split)
        for i in range(num_split):
            if(i == num_split-1):
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:]],x[edge_index[1][N_split * i:]]).cpu()
            else:
                edge_sim1 = F.cosine_similarity(x[edge_index[0][N_split * i:N_split*(i+1)]],x[edge_index[1][N_split * i:N_split*(i+1)]]).cpu()
            # print(edge_sim1)
            edge_sim1 = edge_sim1.cpu()
            edge_sims = torch.cat([edge_sims,edge_sim1])
        # edge_sims = edge_sims.to(device)
    else:
        # calculate edge simlarity
        edge_sims = F.cosine_similarity(x[edge_index[0]],x[edge_index[1]])
    # find dissimilar edges and remote them
    dissim_edges_index = np.where(edge_sims.cpu()<=args.prune_thr)[0]
    edge_weights[dissim_edges_index] = 0
    # select the nodes between dissimilar edgesy
    dissim_edges = edge_index[:,dissim_edges_index]    # output: [[v_1,v_2],[u_1,u_2]]
    dissim_nodes = torch.cat([dissim_edges[0],dissim_edges[1]]).tolist()
    dissim_nodes = list(set(dissim_nodes))
    # update structure
    updated_edge_index = edge_index[:,edge_weights>0.0]
    updated_edge_weights = edge_weights[edge_weights>0.0]
    return updated_edge_index,updated_edge_weights,dissim_nodes 

# def reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x,ori_x,ori_edge_index,device, idx, large_graph=True):
def reconstruct_prune_unrelated_edge(args,poison_edge_index,poison_edge_weights,poison_x, device):
    poison_x = poison_x.to(device)

    # AE = MLPAE(poison_x, poison_x[len(ori_x):], device, args.rec_epochs)
    AE = MLPAE(poison_x, device, args.rec_epochs)
    AE.fit()

    rec_score_ori = AE.inference(poison_x)

    threshold = np.percentile(rec_score_ori.detach().cpu().numpy(), args.ae_thr)
    mask = rec_score_ori>threshold
    abnormal_nodes = torch.nonzero(mask, as_tuple=True)[0].cpu()

    keep_edges_mask = ~(mask[poison_edge_index[0]] | mask[poison_edge_index[1]])
    filtered_poison_edge_index = poison_edge_index[:, keep_edges_mask]
    filtered_poison_edge_weights = poison_edge_weights[keep_edges_mask]
    return filtered_poison_edge_index,filtered_poison_edge_weights, abnormal_nodes



from models.GCN import ConsistencyDetector
import torch
from models.construct import model_construct

def detect_by_detector(args, detector:ConsistencyDetector, x, edge_index, edge_weight):
   
    score, attr_err, homo_err = detector.inference(x, edge_index, edge_weight)

    threshold = np.percentile(score.detach().cpu().numpy(), args.ae_thr)
    mask = score>threshold
    abnormal_nodes = torch.nonzero(mask, as_tuple=True)[0].cpu()
    keep_edges_mask = ~(mask[edge_index[0]] | mask[edge_index[1]])
    filtered_poison_edge_index = edge_index[:, keep_edges_mask]
    filtered_poison_edge_weights = edge_weight[keep_edges_mask]

    return filtered_poison_edge_index,filtered_poison_edge_weights

def detect_and_prune_by_consistency_detector(
    args,
    data,
    x: torch.Tensor,                 # [N, F]
    edge_index: torch.Tensor,         # [2, E]
    edge_weight: torch.Tensor = None,
    device: torch.device = None,
):

    assert device is not None, "Please provide device"

    x = x.to(device)
    edge_index = edge_index.to(device)
    if edge_weight is not None:
        edge_weight = edge_weight.to(device)
    else:
        edge_weight = edge_index.new_ones(edge_index.size(1), dtype=torch.float)

    N = x.size(0)

    gcn_encoder = model_construct(args,args.graph_encoder,data, device).to(device) 
    detector = ConsistencyDetector(
        gcn_encoder=gcn_encoder,
        c=getattr(args, "c", 1),
        a=getattr(args, "a", 1),
        b=getattr(args, "b", 1),
        lr=args.ae_lr,
        epochs=args.rec_epochs,
        device=device,
        add_selfloop_for_mean=True,
        dec_hidden=getattr(args, "hidden", 128),
        dec_dropout=getattr(args, "dropout", 0.0),
        use_sigmoid_x=False,
    ).to(device)


    detector.fit(x, edge_index, edge_weight)
    score, attr_err, homo_err = detector.inference(x, edge_index, edge_weight)

    threshold = np.percentile(score.detach().cpu().numpy(), args.ae_thr)
    mask = score>threshold
    abnormal_nodes = torch.nonzero(mask, as_tuple=True)[0].cpu()

    keep_edges_mask = ~(mask[edge_index[0]] | mask[edge_index[1]])
    filtered_poison_edge_index = edge_index[:, keep_edges_mask]
    filtered_poison_edge_weights = edge_weight[keep_edges_mask]

    global_mean = score.mean()
    global_std = score.std()
    z = (score - global_mean) / (global_std + 1e-12)

    tau = getattr(args, "conf_tau", 1.0)
    gamma = torch.sigmoid(z / tau)  # [N] on device

    return filtered_poison_edge_index,filtered_poison_edge_weights, abnormal_nodes, gamma, detector
   


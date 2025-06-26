import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp,GcnLayers,DGIprompt,GraphCLprompt,Lpprompt
from layers import GCN, AvgReadout 
import tqdm
import numpy as np

# Add device detection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_device(tensor_or_list, target_device):
    """Helper function to move tensors to the target device"""
    if isinstance(tensor_or_list, torch.Tensor):
        return tensor_or_list.to(target_device)
    elif isinstance(tensor_or_list, list):
        return [to_device(item, target_device) for item in tensor_or_list]
    elif isinstance(tensor_or_list, tuple):
        return tuple(to_device(item, target_device) for item in tensor_or_list)
    else:
        return tensor_or_list

def ensure_tensor_on_device(tensor, device):
    """Ensure a tensor is on the specified device"""
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).to(device)
    else:
        return torch.tensor(tensor, device=device)

def check_and_fix_device_mismatch(*args, target_device=None):
    """Check if all tensors are on the same device and fix if needed"""
    if not args:
        return args
    
    # Find the target device from the first tensor
    if target_device is None:
        for arg in args:
            if isinstance(arg, torch.Tensor):
                target_device = arg.device
                break
    
    if target_device is None:
        return args
    
    # Move all tensors to the target device
    fixed_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            fixed_args.append(arg.to(target_device))
        else:
            fixed_args.append(arg)
    
    return tuple(fixed_args)

class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation,sample,a1,a2,a3,a4,num_layers_num,p):
        super(PrePrompt, self).__init__()
        self.dgi = DGI(n_in, n_h, activation)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()

        self.weighted_feature=weighted_feature(a1,a2,a3)

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.dgiprompt = DGIprompt(n_in, n_h, activation)
        self.graphcledgeprompt = GraphCLprompt(n_in, n_h, activation)
        self.lpprompt = Lpprompt(n_in, n_h)
        # Register sample as a buffer so it moves with the model to the correct device
        if isinstance(sample, torch.Tensor):
            self.register_buffer('sample', sample.clone().detach().to(torch.int))
        else:
            self.register_buffer('sample', torch.tensor(sample, dtype=int))

        # self.dffprompt = weighted_prompt(2)
        self.loss = nn.BCEWithLogitsLoss()
        self.act = nn.ELU()

    def forward(self, seq1, seq2, seq3, seq4, seq5, seq6, adj, aug_adj1edge, aug_adj2edge, aug_adj1mask, aug_adj2mask,
                sparse, msk, samp_bias1, samp_bias2,
                lbl):
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        
        # Use the comprehensive device checking function for all tensor inputs
        (seq1, seq2, seq3, seq4, seq5, seq6, adj, aug_adj1edge, aug_adj2edge, 
         aug_adj1mask, aug_adj2mask, msk, samp_bias1, samp_bias2, lbl) = check_and_fix_device_mismatch(
            seq1, seq2, seq3, seq4, seq5, seq6, adj, aug_adj1edge, aug_adj2edge, 
            aug_adj1mask, aug_adj2mask, msk, samp_bias1, samp_bias2, lbl, target_device=device)
        
        # Handle sparse parameter
        if isinstance(sparse, torch.Tensor):
            sparse = sparse.to(device)
        
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        logits1 = self.dgi(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits2 = self.graphcledge(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        
        
        logits4 = self.dgiprompt(self.gcn, seq1, seq2, adj, sparse, msk, samp_bias1, samp_bias2)
        logits5 = self.graphcledgeprompt(self.gcn, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, sparse, msk,
                                   samp_bias1,
                                   samp_bias2, aug_type='edge')
        logits6 = self.lpprompt(self.gcn,seq1,adj,sparse)
        # print("logits1=",logits1)
        # print("logits2=",logits2)
        # print("logits3=",logits3)
        # print("logitssize=",logits3.shape)
        # print("logits1=",logits1)
        # print("logits1size=",logits1.shape)
        # print("lbl",lbl)

        # print("lblsize",lbl.shape)

        logits11 = logits1 + self.a4*logits4
        logits22 = logits2 + self.a4*logits5
        logits33 = logits3 + self.a4*logits6

        # logits11 = self.dffprompt(logits1,logits4)
        # logits22 = self.dffprompt(logits2,logits5)
        # logits33 = self.dffprompt(logits3,logits6)

        dgiloss = self.loss(logits11, lbl)
        graphcledgeloss = self.loss(logits22, lbl)
        lploss = compareloss(logits33,self.sample,temperature=1.5)
        lploss.requires_grad_(True)
        
        # print("promptdgi",self.dgi.prompt)
        # print("gcn",self.gcn.fc.weight)
        # print("promptLP",self.lp.prompt)


        # print("dgiloss",dgiloss)
        # print("graphcl",graphcledgeloss)
        # print("lploss",'{:.8f}'.format(lploss)) 

        # print("a1=", self.a1, "a2=", self.a2,"a3=",self.a3)
        # ret =self.weighted_feature(dgiloss,graphcledgeloss,lploss)
        ret = self.a1 * dgiloss + self.a2 * graphcledgeloss + self.a3 * lploss


        # ret2 = self.a1 * dgilossprompt + self.a2 * graphcledgelossprompt + self.a3 * lplosspropmt
        
        # ret = ret1 +self.a4*ret2

        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        # Ensure all input tensors are on the same device as the model
        device = next(self.parameters()).device
        
        # Use the comprehensive device checking function
        seq, adj, msk, LP = check_and_fix_device_mismatch(seq, adj, msk, LP, target_device=device)
        
        # Handle sparse parameter (could be a boolean or tensor)
        if isinstance(sparse, torch.Tensor):
            sparse = sparse.to(device)
        
        # print("seq",seq.shape)
        # print("adj",adj.shape)
        h_1 = self.gcn(seq, adj, sparse,LP)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

    def to(self, device):
        """Override to method to ensure all components are moved to the target device"""
        super().to(device)
        # Ensure all submodules are moved to the device
        self.dgi = self.dgi.to(device)
        self.graphcledge = self.graphcledge.to(device)
        self.lp = self.lp.to(device)
        self.gcn = self.gcn.to(device)
        self.read = self.read.to(device)
        self.weighted_feature = self.weighted_feature.to(device)
        self.dgiprompt = self.dgiprompt.to(device)
        self.graphcledgeprompt = self.graphcledgeprompt.to(device)
        self.lpprompt = self.lpprompt.to(device)
        self.loss = self.loss.to(device)
        self.act = self.act.to(device)
        return self
    
    def simple_device_check(self):
        """Simple device check that just shows the main model device"""
        try:
            device = next(self.parameters()).device
            print("Model device:", str(device))
            return device
        except Exception as e:
            print("Could not determine model device:", str(e))
            return None
    
    def check_device_consistency(self):
        """Debug method to check if all components are on the same device"""
        print("=== Device Consistency Check ===")
        
        try:
            device = next(self.parameters()).device
            print("Model device:", str(device))
        except StopIteration:
            print("Model has no parameters!")
            return None
        except Exception as e:
            print("Error getting model device:", str(e))
            return None
        
        components = [
            ('dgi', self.dgi),
            ('graphcledge', self.graphcledge),
            ('lp', self.lp),
            ('gcn', self.gcn),
            ('read', self.read),
            ('weighted_feature', self.weighted_feature),
            ('dgiprompt', self.dgiprompt),
            ('graphcledgeprompt', self.graphcledgeprompt),
            ('lpprompt', self.lpprompt),
        ]
        
        for name, component in components:
            try:
                if component is None:
                    print(name + ": Component is None")
                    continue
                
                # Check if it's a PyTorch module
                if not hasattr(component, 'parameters'):
                    print(name + ": Not a PyTorch module")
                    continue
                
                # Safely get parameters
                try:
                    params = list(component.parameters())
                    if params:
                        comp_device = params[0].device
                        print(name + ":", str(comp_device))
                        if str(comp_device) != str(device):
                            print("  Device mismatch:", name, "is on", str(comp_device), "but model is on", str(device))
                    else:
                        print(name + ": No parameters (this is normal for some layers)")
                except Exception as param_error:
                    print(name + ": Could not check parameters -", str(param_error))
                    
            except Exception as e:
                print(name + ": Error -", str(e))
        
        print("=== End Device Check ===")
        return device




def mygather(feature, index):
    # Ensure index is on the same device as feature and has correct dtype
    index = index.to(feature.device).long()  # Convert to int64 dtype
    
    # print("index",index)
    # print("indexsize",index.shape)  
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))
    # print(tuples)

    # print("feature",feature)
    # print("featuresize",feature.shape)
    # print("index",index)
    # print("indexsize",index.shape)
    res = torch.gather(feature, dim=0, index=index)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):
    # Ensure all tensors are on the same device
    device = feature.device
    tuples = tuples.to(device).long()  # Ensure correct dtype
    
    # print("feature",feature)
    # print("tuple",tuples)
    # feature=feature.cpu()
    # tuples = tuples.cpu()
    h_tuples=mygather(feature,tuples)
    # print("tuples",h_tuples)
    # Fix: Create temp tensor on the same device as tuples with correct dtype
    temp = torch.arange(0, len(tuples), device=tuples.device, dtype=torch.long)
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    # temp = m(temp)
    # temp=temp.cuda()
    h_i = mygather(feature, temp)
    # print("h_i",h_i)
    # print("h_tuple",h_tuples)
    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj,n):
    nodenum=adj.shape[0]
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    print("#############")
    print("start sampling disconnected tuples")
    for i in tqdm.trange(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    # Convert to tensor with correct dtype (int64/long)
    return torch.tensor(res, dtype=torch.long)


class weighted_feature(nn.Module):
    def __init__(self,a1,a2,a3):
        super(weighted_feature, self).__init__()
        self.weight= nn.Parameter(torch.FloatTensor(1,3), requires_grad=True)
        self.reset_parameters(a1,a2,a3)
    def reset_parameters(self,a1,a2,a3):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(a1)
        self.weight[0][1].data.fill_(a2)
        self.weight[0][2].data.fill_(a3)
    def forward(self, graph_embedding1,graph_embedding2,graph_embedding3):
        print("weight",self.weight)
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2 + self.weight[0][2] * graph_embedding3
        return graph_embedding
    
    def to(self, device):
        """Override to method to ensure proper device placement"""
        super().to(device)
        return self
    




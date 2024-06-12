import mindspore as ms
from mindspore import nn, Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops
import numpy as np
# from audtorch.metrics.functional import pearsonr
l2_normalize = ops.L2Normalize(axis=-1)
fill_diagonal = ops.FillDiagonal(0.)
class Loss(nn.Cell):
    def __init__(self, t, Nlabel):
        super(Loss, self).__init__()

        self.Nlabel = Nlabel
        self.t = t




    
    def label_graph2(self, emb, inc_labels, inc_L_ind):
        # label guide the embedding feature
        cls_num = inc_labels.shape[-1]
        valid_labels_sum = ops.matmul(inc_L_ind.float(), inc_L_ind.float().T) #[n, n] 

        # graph = torch.matmul(inc_labels, inc_labels.T).fill_diagonal_(0)
        
        graph = (ops.matmul(inc_labels, inc_labels.T).mul(valid_labels_sum) / (ops.matmul(inc_labels, inc_labels.T).mul(valid_labels_sum)+100))
        eye = mnp.eye(graph.shape[0])
        graph = graph.masked_fill(eye==1,0.)
        # print((graph>0.1).sum(),graph.shape)

        graph = ops.clamp(graph,min=0,max=1.)
        emb = l2_normalize(emb)
        # graph = graph.mul(graph>0.2)
        # graph = (inc_labels.mm(inc_labels.T))
        # graph = 0.5*(graph+graph.t())¸
        
        loss = 0
        Lap_graph  = ops.diag(graph.sum(1))- graph
        loss = ops.trace(emb.t().mm(Lap_graph).mm(emb))/emb.shape[0]
        return loss/emb.shape[0] #loss/number of views

    def forward_contrast(self, si, vi, wei):
        ## S1 S2 [v d]
        si = si[wei.bool()]
        vi = vi[wei.bool()]
        n = si.shape[0]
        N = 2 * n
        if n <= 1:
            return 0
        si = l2_normalize(si)
        vi = l2_normalize(vi)
        if si.shape[0]<=1 and vi.shape[0]<=1:
            return 0

        svi = ops.cat((si, vi), axis=0)

        sim = ops.matmul(svi, svi.T)
        # sim = (sim/self.t).exp()
        # print(sim)
        eye = mnp.eye(N)
        pos_mask = ops.zeros((N, N))
        pos_mask[:n,:n] = ops.ones((n, n))
        neg_mask = 1-pos_mask
        pos_mask = pos_mask.masked_fill(eye==1,0)
        neg_mask = neg_mask.masked_fill(eye==1,0)
        pos_pairs = sim.masked_select(pos_mask.bool())
        neg_pairs = sim.masked_select(neg_mask.bool())
        # prop = torch.exp(pos_pairs).mean()/(torch.exp(pos_pairs).mean()+torch.abs(torch.exp(neg_pairs)).mean())
        # loss = -torch.log(prop)
        loss = (neg_pairs).square().mean()/(((pos_pairs+1+1e-6)/2).mean())
        # loss = (neg_pairs).square().mean()/(pos_pairs).square().mean()
        # target = torch.eye(N,device=sim.device)
        # target[:n,:n] = torch.ones((n, n),device=sim.device)
        # loss = (-target.mul(torch.log((sim+1)/2+1e-6))-(1-target).mul(torch.log(1-sim.square()+1e-6))).mean()


        # assert sum(int(ops.isnan(loss))) == 0
        return loss/2



    def wmse_loss(self,input, target, weight, reduction='mean'):
        ret = (ops.diag(weight).mm(target - input)) ** 2
        ret = ops.mean(ret)
        return ret

    def cont_loss(self,S,V,inc_V_ind):
        loss_Cont = 0
        if isinstance(S,list):
            S = ops.stack(S,1) #[n v d]

        if isinstance(V,list):
            V = ops.stack(V,1) #[n v d]
        for i in range(S.shape[0]):
            loss_Cont += self.forward_contrast(S[i], V[i], inc_V_ind[i,:])
        return loss_Cont







    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):

        # assert ops.sum(int(ops.isnan(ops.log(target_pre)))) == 0
        # assert ops.sum(int(ops.isnan(ops.log(1 - target_pre + 1e-5)))) == 0
        res=ops.abs((sub_target.mul(ops.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(ops.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return ops.sum(res)/ops.sum(inc_L_ind)
        elif reduction=='sum':
            return ops.sum(res)
        elif reduction=='none':
            return res



    

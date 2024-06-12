# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import mindspore as ms
from mindspore import nn, Tensor
import mindspore.ops as ops
import mindspore.numpy as mnp
isnan = ops.IsNan()
mul = ops.Mul()
normalize = ops.L2Normalize(-1)
class Loss(nn.Cell):
    def __init__(self):
        super(Loss, self).__init__()

    def contrastive_loss2(self, x, inc_labels, inc_V_ind, inc_L_ind):
        n = x.shape[0]
        v = x.shape[1]

        if n == 1:
            return 0
        valid_labels_sum = ops.matmul(inc_L_ind.float(), inc_L_ind.float().T) #[n, n] 

        labels = (ops.matmul(inc_labels, inc_labels.T) / (valid_labels_sum + 1e-9))
        eye = mnp.eye(n)
        labels = ops.masked_fill(labels,eye==1,0)
        # labels = torch.softmax(labels.masked_fill(labels==0,-1e9),dim=-1)
        x = normalize(x)
        x = x.transpose(1,0,2) #[v,n,d]
        x_T = x.transpose(0,2,1)#[v,d,n]
        sim = (1+ops.matmul(x,x_T))/2 # [v, n, n]
        mask_v = mul((inc_V_ind.T).unsqueeze(-1),((inc_V_ind.T).unsqueeze(1))) #[v, n, n]
        mask_v = mask_v.masked_fill(eye==1,0.)


        loss = self.weighted_BCE_loss(sim.view(v,-1),labels.view(1,n*n)+mnp.zeros([v,n*n],dtype=ms.float32),mask_v.view(v,-1),reduction='none')
        # assert torch.sum(torch.isnan(loss)).item() == 0
        
        loss =loss.sum(axis=-1)/(mask_v.view(v,-1).sum(axis=-1))
        return 0.5*loss.sum()/v


    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert (isnan(ops.log(target_pre))).sum() == 0
        assert (isnan(ops.log(1 - target_pre + 1e-5))).sum() == 0
        res=ops.abs((mul(sub_target,ops.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(ops.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return (res).sum()/(inc_L_ind).sum()
        elif reduction=='sum':
            return (res).sum()
        elif reduction=='none':
            return res
                            
    # def BCE_loss(self,target_pre,sub_target):
    #     return torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
    #                                     + (1-sub_target).mul(torch.log(1 - target_pre + 1e-10)))))
    
    
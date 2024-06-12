import mindspore as ms
from mindspore import nn, Tensor
import mindspore.numpy as mnp
import mindspore.ops as ops
import numpy as np
import time
matmul=ops.MatMul()
class Loss(nn.Cell):
    def __init__(self, t):
        super(Loss, self).__init__()
        self.t = t

        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")

    def contrast_loss(self, v1, v2, we1, we2):
        t1=time.time()
        normalize = ops.L2Normalize(-1)
        mask_miss_inst = we1.mul(we2).numpy() # mask the unavailable instances
        if (mask_miss_inst==1).sum()==0:
            return 0
        mask_miss_inst = Tensor(np.where(mask_miss_inst))
        mask_miss_inst=mask_miss_inst.squeeze(0)
        
        a1= v1[mask_miss_inst]
        a2= v2[mask_miss_inst]
        t2=time.time()

        # mask_miss_inst = we1.mul(we2).numpy()
        # print(a1.shape[0])
        if a1.shape[0]<4:
            return 0
        a1 = normalize(a1) #normalize two vectors
        a2 = normalize(a2)
    
        n = a1.shape[0]
        N = 2 * n
        if n == 0:
            return 0

        h = ops.concat((a1, a2), axis=0)
        sim_mat = matmul(h, h.T) / self.t
        for i in range(N):
            sim_mat[i,i] = 0
        label1 = Tensor(np.array(range(n,N),dtype=np.int32))
        label2 = Tensor(np.array(range(0,n),dtype=np.int32))
        label = ops.concat((label1,label2),axis=0)
        # print(label)
        loss = self.criterion(sim_mat, label)

        return loss/2
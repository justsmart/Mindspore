

import copy
import math


import mindspore as ms
from mindspore import nn, Tensor,Parameter
import mindspore.numpy as mnp
import mindspore.ops as ops
from transformer import Trans
# matmul = ops.MatMul()
relu = nn.ReLU()
mul = ops.Mul()
isnan = ops.IsNan()
stack1=ops.Stack(axis=1)
def get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])
    
def setEmbedingModel(d_list,d_out):
    
    return nn.CellList([Mlp(d,d,d_out)for d in d_list])


class Mlp(nn.Cell):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Dense(in_dim, mlp_dim)
        self.fc2 = nn.Dense(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(p=dropout_rate)
            self.dropout2 = nn.Dropout(p=dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def construct(self, x):

        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out



class Model(nn.Cell):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, classes_num, dropout,exponent=2):
        super().__init__()
        # self.Trans = TransformerWoDecoder(input_len, d_model, n_layers, heads, dropout)
        self.Trans = Trans(embed_dim=d_model,num_layers=n_layers,num_heads=heads)
        self.embeddinglayers = setEmbedingModel(d_list,d_model)
        self.view_num = input_len

        self.CFTrans = Trans(embed_dim=d_model,num_layers=n_layers,num_heads=heads)
        self.classifiers = nn.CellList([nn.Dense(d_model,1)for _ in range(classes_num)])
        self.classifier2 = nn.Dense(d_model,classes_num)
        self.adaptive_weighting = nn.Dense(d_model,1,has_bias=False)
        self.weights = Parameter(ops.softmax(mnp.randn([1,self.view_num,1]),axis=1))
        self.exponent = exponent
        self.cls_tokens = Parameter(mnp.randn(1, classes_num, d_model))
        self.classes_num = classes_num
        self.d_model = d_model
        # torch.nn.init.xavier_uniform_(self.cls_tokens,gain=1)
        
    def construct(self,x,mask=None,label_mask = None):
        B = mask.shape[0]
        
        for i in range(self.view_num):
            x[i] = self.embeddinglayers[i](x[i])
        x = stack1(x) # B,view,d
        x = self.Trans(x,mask)
        x_tran = x
        x_weighted = ops.pow(self.weights.expand_as(x),self.exponent)
        x_weighted_mask = ops.softmax(x_weighted.masked_fill(mask.unsqueeze(2)==0, -1e9),axis=1) #[B, self.view_num, d_e]
        assert mnp.sum(isnan(x_weighted_mask)) == 0
        # print('mask',mask[:3,:])
        x = mul(x,x_weighted_mask)
        
        fusion_x = x.sum(-2)
        
        cls_tokens = self.cls_tokens.expand_as(mnp.randn(B, self.classes_num, self.d_model))
        x_tokens = ops.concat((fusion_x.unsqueeze(1),cls_tokens),axis=1)
        if label_mask is not None:
            label_mask = ops.concat((mnp.ones([B,1]),label_mask),axis=1)  
        else: 
            label_mask = None
        x_tokens = self.CFTrans(x_tokens)
        pred2 = [ops.sigmoid(classifier(x_tokens[:,i+1])) for i,classifier in enumerate(self.classifiers)]
        pred2 = stack1(pred2).squeeze(-1)
        pred = ops.sigmoid(self.classifier2(x_tokens[:,0]))
        # print(pred[0].dtype)
        return pred,pred2,x_tran,None


def get_model(input_len, d_list,d_model=768,n_layers=2,heads=4,classes_num=10,dropout=0.2,exponent=1,load_weights=None):
    
    assert d_model % heads == 0
    assert dropout < 1

    model = Model(input_len, d_model, n_layers, heads, d_list, classes_num, dropout, exponent)

    # if load_weights is not None:
        # print("loading pretrained weights...")
        # model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    # else:
        # for p in model.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p) 
        # pass
    
    
    # model = model.to(device)
    
    return model
    
if __name__=='__main__':
    import mindspore.context as context
    context.set_context(device_target="GPU")
    inp = [Tensor(mnp.ones([10,15]),ms.float32),Tensor(mnp.ones([10,20]),ms.float32)]

    a = Tensor([0.2,0.3,1],ms.float32)

    we = Tensor(mnp.ones([10,2]),ms.float32)
    net = get_model(2, [15,20])
    oup=net(inp,we)
    print(oup[1].shape)
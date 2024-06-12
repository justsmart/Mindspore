
import copy
import math
import mindspore as ms
from mindspore import nn, Tensor,Parameter
import mindspore.numpy as mnp
import mindspore.ops as ops
from transformer import Trans
relu = nn.ReLU()
mul = ops.Mul()
isnan = ops.IsNan()
stack1=ops.Stack(axis=1)

def get_clones(module, N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])
    
def setEmbedingModel(d_list,d_out):
    return nn.CellList([nn.Dense(d,d_out) for d in d_list])

def setReEmbedingModel(d_list,d_out):
    return nn.CellList([nn.Dense(d_out,d)for d in d_list])


class Model(nn.Cell):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, classes_num, dropout,recover=True):
        super().__init__()
        # self.ETrans = TransformerWoDecoder(input_len,d_model, n_layers, heads, dropout)
        # self.DTrans = TransformerWoDecoder(input_len,d_model, n_layers, heads, dropout)
        self.ETrans = Trans(embed_dim=d_model,input_channels=input_len,num_layers=n_layers,num_heads=heads)
        self.DTrans = Trans(embed_dim=d_model,input_channels=input_len,num_layers=n_layers,num_heads=heads)
        self.embeddinglayers = setEmbedingModel(d_list,d_model)
        self.re_embeddinglayers = setReEmbedingModel(d_list,d_model)
        self.recover = recover
        # self.classifier = nn.Linear(d_model,classes_num)
    def construct(self,data,mask=None):
        view_num = len(data)
        x = [None]*view_num
        for i in range(view_num): # encode input view to features with same dimension 
            x[i] = self.embeddinglayers[i](data[i])
        x = stack1(x) # B,view,d
        if mask==None:
            mask = Tensor(mnp.ones(x.shape[:2]),dtype=ms.float32)
        
        x = self.ETrans(x,mask)
        encX = x
        # H = torch.einsum('bvd->bd',H)
        x = mul(x,mask.unsqueeze(2))
        x = x.sum(-2)/mask.sum(axis=-1,keepdims=True)

        H = x
        # ori_x = x.detach().clone()

        x = x.unsqueeze(1).repeat(view_num,axis=1) #[b v d]
        x = self.DTrans(x,None)
        
        decX = x
        # H[(1-mask).bool()] = x[(1-mask).bool()].clone().detach() 
        
        x_bar = [None]*view_num
        for i in range(view_num):       
            x_bar[i] = self.re_embeddinglayers[i](x[:,i])
        
        # x_bar = torch.stack(x_bar,dim=1) # B,view,d
        # print(x_bar[0][0,:].detach().cpu().numpy().shape)
        return encX,decX,x_bar,H,None,None


def get_model( d_list,d_model=768,n_layers=2,heads=4,classes_num=10,dropout=0.2,load_weights=None):
    
    assert d_model % heads == 0
    assert dropout < 1

    # model = Transformer(input_len, output_len, d_model, n_layers, heads, dropout)
    model = Model(len(d_list), d_model, n_layers, heads, d_list, classes_num, dropout)

    # if load_weights is not None:
    #     print("loading pretrained weights...")
    #     # model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    # else:
    #     for p in model.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p) 
    
    
    # model = model.to(device)
    
    return model




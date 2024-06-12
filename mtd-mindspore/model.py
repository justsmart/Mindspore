import copy
import math


import mindspore as ms
from mindspore import nn, Tensor,Parameter
import mindspore.numpy as mnp
import mindspore.ops as ops

relu = nn.ReLU()
mul = ops.Mul()
isnan = ops.IsNan()
stack1=ops.Stack(axis=1)

# mean=ReduceMean()
class encoder(nn.Cell):
    def __init__(self, n_dim, dims, n_z):
        super(encoder, self).__init__()
        # print(n_dim,dims[0])
        self.enc_1 = nn.Dense(n_dim, dims[0])
        self.enc_2 = nn.Dense(dims[0], dims[1])
        self.enc_3 = nn.Dense(dims[1], dims[2])
        self.z_layer = nn.Dense(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)
        self.relu = nn.ReLU()
    def construct(self, x):
        enc_h1 = self.relu(self.enc_1(x))
        enc_h2 = self.relu(self.enc_2(enc_h1))
        enc_h3 = self.relu(self.enc_3(enc_h2))
        z = self.z_b0(self.z_layer(enc_h3))
        return z
class decoder(nn.Cell):
    def __init__(self, n_dim, dims, n_z):
        super(decoder, self).__init__()
        self.dec_0 = nn.Dense(n_z, n_z)
        self.dec_1 = nn.Dense(n_z, dims[2])
        self.dec_2 = nn.Dense(dims[2], dims[1])
        self.dec_3 = nn.Dense(dims[1], dims[0])
        self.x_bar_layer = nn.Dense(dims[0], n_dim)
        self.relu = nn.ReLU()
    def construct(self, z):
        r = self.relu(self.dec_0(z))
        dec_h1 = self.relu(self.dec_1(r))
        dec_h2 = self.relu(self.dec_2(dec_h1))
        dec_h3 = self.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
        return x_bar

class net(nn.Cell):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(net, self).__init__()
        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

        self.encoder_list = nn.CellList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.decoder_list = nn.CellList([decoder(n_input[i], dims[i], 1*n_z) for i in range(len(n_input))])
        self.encoder2_list = nn.CellList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])

        self.regression = nn.Dense(n_z, nLabel)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()
        self.nLabel = nLabel
        # self.BN = nn.BatchNorm1d(n_z)

    def construct(self, mul_X, we,mode,sigma):
        we = we.float()
        batch_size = mul_X[0].shape[0]
        summ = 0
        prop = sigma
        share_zs = []
        if mode =='train':
            for i,X in enumerate(mul_X):

                mask_len = int(prop*X.shape[-1])

                st = ops.randint(low=0,high=X.shape[-1]-mask_len-1,size=(X.shape[0],))
                # print(st,st+mask_len)
                mask = ops.ones_like(X)
                for j,e in enumerate(mask): 
                    mask[j,st[j]:st[j]+mask_len] = 0
                mul_X[i] = mul(mul_X[i],mask)

                # for s in range(mul_X[i].size(0)):
                #     mask = sample(range(X.size(-1)),mask_len)
                #     mul_X[i][s,mask] = 0
                
        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(mul_X[enc_i])
            share_zs.append(z_i)
            summ += ops.diag(we[:, enc_i]).mm(z_i)
        wei = 1 / ops.sum(we, 1)
        s_z = ops.diag(wei).mm(summ)
        
        summvz = 0
        viewsp_zs = []
        for enc_i, enc in enumerate(self.encoder2_list):
            z_i = enc(mul_X[enc_i])
            viewsp_zs.append(z_i)
            summvz += ops.diag(we[:, enc_i]).mm(z_i)
        wei = 1 / ops.sum(we, 1)
        v_z = ops.diag(wei).mm(summvz)
        
        # z = torch.cat((s_z,v_z),-1)
        z = mul(s_z,ops.sigmoid(v_z))
        # z = self.BN(z)
        z = self.relu(z)
        # z = s_z+v_z

        x_bar_list = []
        for dec_i, dec in enumerate(self.decoder_list):
            x_bar_list.append(dec(share_zs[dec_i]+viewsp_zs[dec_i]))
            # x_bar_list.append(dec(viewsp_zs[dec_i]))
            # x_bar_list.append(dec(F.sigmoid(s_z).mul(viewsp_zs[dec_i])))
        
        
        logi = self.regression(z) #[n c]
        # logi = self.labelgcn(dep_graph,logi.T).T
        
        # logi = F.relu(z).mm(W.T)
        yLable = self.act(logi)
        return x_bar_list, yLable, z, share_zs, viewsp_zs

def get_model(n_stacks,n_input,n_z,Nlabel):
    model = net(n_stacks=n_stacks,n_input=n_input,n_z=n_z,nLabel=Nlabel)
    return model
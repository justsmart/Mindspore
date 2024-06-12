from mindspore import nn, ops
import mindspore as ms
from typing import Optional, Dict
from mindspore.common.initializer import Normal
from mindspore.common.initializer import initializer
from mindspore import Parameter, Tensor
import mindspore.numpy as mnp
import copy
class Attention(nn.Cell):
    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = ms.Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(p=1.0-attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(p=1.0-keep_prob)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x,mask=None):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = ops.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = ops.unstack(qkv, axis=0)
        attn = self.q_matmul_k(q, k)
        attn = ops.mul(attn, self.scale)
        if mask is not None:
            mask = mask.unsqueeze(1).float()
            mask = ops.matmul(mask.unsqueeze(-1),mask.unsqueeze(-2))#mask shape is [bs 1 view view]

            # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
            attn = ops.masked_fill(attn,mask == 0, -1e9)
    
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = self.attn_matmul_v(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out





class FeedForward(nn.Cell):
    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(p=1.0-keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x


class ResidualCell(nn.Cell):
    def __init__(self, cell):
        super(ResidualCell, self).__init__()
        self.cell = cell

    def construct(self, x):
        """ResidualCell construct."""
        return self.cell(x) + x

def getcloneLayers(module,N):
    return nn.CellList([copy.deepcopy(module) for i in range(N)])
class TransformerEncoder(nn.Cell):
    def __init__(self,
                 dim: int,
                #  num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        self.normalization1 = norm((dim,))
        self.normalization2 = norm((dim,))
        self.attention = Attention(dim=dim,
                                num_heads=num_heads,
                                keep_prob=keep_prob,
                                attention_keep_prob=attention_keep_prob)

        self.feedforward = FeedForward(in_features=dim,
                                    hidden_features=mlp_dim,
                                    activation=activation,
                                    keep_prob=keep_prob)

        # layers.append(
        #     nn.SequentialCell([
        #         ResidualCell(nn.SequentialCell([normalization1, attention])),
        #         ResidualCell(nn.SequentialCell([normalization2, feedforward]))
        #     ])
        # )
        # self.layers = nn.SequentialCell(layers)

    def construct(self, x, mask=None):
        """Transformer construct."""
        x1 = self.normalization1(x)
        x = x+self.attention(x1,mask)
        x2 = self.normalization2(x)
        x = x+self.feedforward(x2)
        return x


def init(init_type, shape, dtype, name, requires_grad):
    """Init."""
    initial = initializer(init_type, shape, dtype).init_data()
    return Parameter(initial, name=name, requires_grad=requires_grad)


class Trans(nn.Cell):
    def __init__(self,
                #  image_size: int = 224,
                 input_channels: int = 3,
                 embed_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 15,
                 mlp_dim: int = 768,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: Optional[nn.Cell] = nn.LayerNorm,
                 classes_num : int =100
                #  pool: str = 'cls'
                 ) -> None:
        super(Trans, self).__init__()

        # self.patch_embedding = PatchEmbedding(image_size=image_size,
        #                                       patch_size=patch_size,
        #                                       embed_dim=embed_dim,
        #                                       input_channels=input_channels)
        # num_patches = self.patch_embedding.num_patches
        self.classes_num = classes_num
        # self.cls_token = init(init_type=Normal(sigma=1.0),
        #                       shape=(1, 1, embed_dim),
        #                       dtype=ms.float32,
        #                       name='cls',
        #                       requires_grad=True)

        # self.pos_embedding = init(init_type=Normal(sigma=1.0),
        #                           shape=(1, num_patches + 1, embed_dim),
        #                           dtype=ms.float32,
        #                           name='pos_embedding',
        #                           requires_grad=True)

        # self.pool = pool
        self.pos_dropout = nn.Dropout(p=1.0-keep_prob)
        self.norm = norm((embed_dim,))
        self.layers = getcloneLayers(TransformerEncoder(dim=embed_dim,
                                              num_heads=num_heads,
                                              mlp_dim=mlp_dim,
                                              keep_prob=keep_prob,
                                              attention_keep_prob=attention_keep_prob,
                                              drop_path_keep_prob=drop_path_keep_prob,
                                              activation=activation,
                                              norm=norm),num_layers)

        self.dropout = nn.Dropout(p=1.0-keep_prob)
        # self.dense = nn.Dense(embed_dim, 11)

    def construct(self, x,mask=None):
        """ViT construct."""
        # x = self.patch_embedding(x)
        # cls_tokens = ops.tile(self.cls_token.astype(x.dtype), (x.shape[0], 1, 1))
        # x = ops.concat((cls_tokens, x), axis=1)
        # x += self.pos_embedding

        # x = self.pos_dropout(x)
        for i in range(len(self.layers)):
            x = self.layers[i](x, mask)
        x = self.norm(x)
        # x = x[:, 0]
        if self.training:
            x = self.dropout(x)
        # x = self.dense(x) # classificaiton layer
        
        return x
        
if __name__=='__main__':
    import mindspore.context as context
    context.set_context(device_target="GPU")
    inp = Tensor(mnp.randn([10,20,200]),ms.float32)

    a = Tensor([0.2,0.3,1],ms.float32)
    net = Trans(embed_dim=200,input_channels=20,num_layers=6,num_heads=4)
    we = Tensor(mnp.ones([10,20]),ms.float32)
    # net = get_model(2, [15,20])
    oup=net(inp,we)
    print(oup.shape)
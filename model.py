import math
from functools import partial
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class GraphConvolution(nn.Module):
    """
    GCN
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class AU_GCN(nn.Module):
    """
       AU-GCN
    """
    def __init__(self, nfeat, nhid, nclass, dropout,adj,mask):
        super(AU_GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.A = nn.Parameter(adj)
        self.mask=mask
    def forward(self, x):
        tmp=[]
        adj = torch.mul(self.A, self.mask)
        adj = adj.to(device)
        adj= torch.sigmoid(adj)
        adj = torch.round(adj)
        for j in range(x.shape[0]):
            t = F.relu(self.gc1(x[j], adj))
            t = self.gc2(t, adj)
            tmp.append(t)
        x=torch.stack(tmp,dim=0)
        x=x.view(x.size(0), -1)
        return x

def token_overlap(x):
    """
    temporal overlap model:here stride=1,ratio=1/6
    """
    x_head=torch.unsqueeze(x[:,0,:],dim=1)
    x_tail=torch.unsqueeze(x[:,-1,:],dim=1)
    x_body=x[:,1:-1,:]
    x_1=torch.cat((x_body,x_tail,x_tail),dim=1)
    x_3=torch.cat((x_head,x_head,x_body),dim=1)
    x_1=x_1[:,:,0:192]
    x_3 = x_3[:, :, 0:192]
    x=torch.cat((x_1,x,x_3),dim=2)
    return x

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedForConvVit(nn.Module):
    """
        temporal overlap model:here stride=1,ratio=1/6
    """
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=1600, norm_layer=None,batch_size=4):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = 30
        self.batch_size=batch_size
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.lin=nn.Linear(1536, 1152)
        self.conv2 = CNNnet2()

    def forward(self, x):
        seq=[]
        batch=len(x)
        for i in range(0, self.num_patches):
            tmp=[]
            for j in range(0,batch):
                tmp.append(x[j][i])
            tmp=torch.stack(tmp,dim=0)
            tmp2 = self.conv2(tmp)
            seq.append(tmp2)
        x=torch.stack(seq,dim=1)
        x=x.view(batch,self.num_patches,-1)
        x=token_overlap(x)
        x = self.norm(x)
        x=self.lin(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., is_LSA=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                       attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.attn =Attention(dim, num_patches=30, heads = num_heads, dim_head = dim//num_heads, dropout = attn_drop_ratio, is_LSA=True)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class SE_Block(torch.nn.Module):
    """
    Squeeze-and-Excitation block
    """
    def __init__(self, ch_in=128, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)				# 全局自适应池化
        self.fc = nn.Sequential(
            torch.nn.Linear(ch_in, ch_in // reduction, bias=False),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(ch_in // reduction, ch_in, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CNNnet2(torch.nn.Module):
    """
        CNN for single frame feature extraction
    """
    def __init__(self):
        super(CNNnet2, self).__init__()
        self.se=SE_Block()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=7,
                            stride=7,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=5,
                            stride=2,
                            padding=1
                            ),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2,stride=2,)
        )


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv3(x)
        x=self.se(x)

        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=1, num_classes=1000,
                 embed_dim=576, depth=12, num_heads=16, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbedForConvVit, norm_layer=None,
                 act_layer=None,batch_size=8,A=None,mask=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.batch_size=batch_size
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = 30
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        self.gcn=AU_GCN(nfeat=1,
                    nhid=4,
                    nclass=8,
                    dropout=0,
                    adj=A,
                    mask=mask)
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.has_logits = True
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # Weight init
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)
        self.gcnlin=nn.Linear(4080,100)
        self.mergelin=nn.Linear(1252,1152)
    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x,y):
        x = self.forward_features(x)
        y=self.gcn(y)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            y=self.gcnlin(y)
            x=torch.cat((x,y),dim=1)
            x=self.mergelin(x)
            x = self.head(x)
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def dual_stream(num_classes: int = 21843, has_logits: bool = True,batchsize:int=8,A=None,mask=None):

    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=1152,
                              depth=4,
                              num_heads=16,
                              representation_size=768 if has_logits else None,
                              num_classes=num_classes,
                              batch_size=batchsize,
                              A=A,
                              mask=mask)
    return model




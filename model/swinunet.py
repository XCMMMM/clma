import torch
import torch.nn as nn
import torch.nn.functional as func
from functools import partial
from einops import rearrange
from typing import Optional


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        x = x.div(keep_prob) * random_tensor
        return x


class PatchEmbedding(nn.Module):
    # 把 图片变成 patch
    def __init__(self, patch_size: int = 4, in_c: int = 3, embed_dim: int = 96, norm_layer: nn.Module = None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=(
            patch_size,) * 2, stride=(patch_size,) * 2)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def padding(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = func.pad(x, (0, self.patch_size - W % self.patch_size,
                             0, self.patch_size - H % self.patch_size,
                             0, 0))
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.proj(x)
        x = rearrange(x, 'B C H W -> B H W C')
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    #  下采样
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    @staticmethod
    def padding(x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        if H % 2 == 1 or W % 2 == 1:
            x = func.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        return x

    @staticmethod
    def merging(x: torch.Tensor) -> torch.Tensor:
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return x

    def forward(self, x):
        x = self.padding(x)
        x = self.merging(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchExpanding(nn.Module):
    #  上采样
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(PatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // 2)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=2, P2=2)
        x = self.norm(x)
        return x


class FinalPatchExpanding(nn.Module):
    def __init__(self, dim: int, norm_layer=nn.LayerNorm):
        super(FinalPatchExpanding, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        x = self.expand(x)
        x = rearrange(x, 'B H W (P1 P2 C) -> B (H P1) (W P2) C', P1=4, P2=4)
        x = self.norm(x)
        return x


class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None,
                 act_layer=nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self,
                 dim: int,
                 window_size: int,
                 num_heads: int,
                 qkv_bias: Optional[bool] = True,
                 attn_drop: Optional[float] = 0.,
                 proj_drop: Optional[float] = 0.,
                 shift: bool = False):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        if shift:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

        coords_size = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_size, coords_size]))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = coords_flatten[:, :,None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index",relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def window_partition(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        x = rearrange(x, 'B (Nh Mh) (Nw Mw) C -> (B Nh Nw) Mh Mw C',Nh=H // self.window_size, Nw=W // self.window_size)
        return x

    def create_mask(self, x: torch.Tensor) -> torch.Tensor:
        _, H, W, _ = x.shape

        assert H % self.window_size == 0 and W % self.window_size == 0, "H or W is not divisible by window_size"

        img_mask = torch.zeros((1, H, W, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self.window_partition(img_mask)
        mask_windows = mask_windows.view(-1,self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)

        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        _, H, W, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -
                           self.shift_size), dims=(1, 2))
            mask = self.create_mask(x)
        else:
            mask = None

        x = self.window_partition(x)
        Bn, Mh, Mw, _ = x.shape
        x = rearrange(x, 'Bn Mh Mw C -> Bn (Mh Mw) C')
        qkv = rearrange(
            self.qkv(x), 'Bn L (T Nh P) -> T Bn Nh L P', T=3, Nh=self.num_heads)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bn // nW, nW, self.num_heads,
                             Mh * Mw, Mh * Mw) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, Mh * Mw, Mh * Mw)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = rearrange(x, 'Bn Nh (Mh Mw) C -> Bn Mh Mw (Nh C)', Mh=Mh)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = rearrange(x, '(B Nh Nw) Mh Mw C -> B (Nh Mh) (Nw Mw) C',
                      Nh=H // Mh, Nw=H // Mw)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size,
                           self.shift_size), dims=(1, 2))
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, shift=False, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    attn_drop=attn_drop, proj_drop=drop, shift=shift)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x_copy = x
        x = self.norm1(x)

        x = self.attn(x)
        x = self.drop_path(x)
        x = x + x_copy

        x_copy = x
        x = self.norm2(x)

        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + x_copy
        return x


class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x).permute(0, 3, 1, 2))))).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.norm2(x).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        return x
    

class BasicBlock(nn.Module):
    def __init__(self,
                 index: int,
                 embed_dim: int = 96,
                 window_size: int = 7,
                 depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 patch_merging: bool = True):

        super(BasicBlock, self).__init__()
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item()
               for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])

        self.blocks_conv = nn.ModuleList([
            CBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        
        if patch_merging:
            self.downsample = PatchMerging(dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for layer in self.blocks:
            x1 = layer(x)
        for conv_layer in self.blocks_conv:
            x2 = conv_layer(x)
        x = x1 + x2
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class BasicBlockUp(nn.Module):
    def __init__(self,
                 index: int,
                 embed_dim: int = 96,
                 window_size: int = 7,
                 depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path: float = 0.1,
                 patch_expanding: bool = True,
                 norm_layer=nn.LayerNorm):
        super(BasicBlockUp, self).__init__()
        index = len(depths) - index - 2
        depth = depths[index]
        dim = embed_dim * 2 ** index
        num_head = num_heads[index]

        dpr = [rate.item() for rate in torch.linspace(0, drop_path, sum(depths))]
        drop_path_rate = dpr[sum(depths[:index]):sum(depths[:index + 1])]

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_head,
                window_size=window_size,
                shift=False if (i % 2 == 0) else True,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if patch_expanding:
            self.upsample = PatchExpanding(
                dim=embed_dim * 2 ** index, norm_layer=norm_layer)
        else:
            self.upsample = nn.Identity()

        self.blocks_conv = nn.ModuleList([
            CBlock(
                dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        
    def forward(self, x):
        for layer in self.blocks:
            x1 = layer(x)
        for conv_layer in self.blocks_conv:
            x2 = conv_layer(x)
        x = x1 + x2
        x = self.upsample(x)
        return x


class SwinUnetEncoder(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 window_size: int = 7,
                 depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm: bool = True):
        super(SwinUnetEncoder, self).__init__()
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer

        #  把图片变成patch
        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,norm_layer=norm_layer if patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = self.build_layers()
        self.down1 = PatchMerging(dim=embed_dim * 2 ** 1, norm_layer=norm_layer)
        self.down2 = PatchMerging(dim=embed_dim * 2 ** 2, norm_layer=norm_layer)

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers

    def forward(self, x):

        #  把图片变成patch [b,c,h,w]-->[b,h/patch_size,w/patch_size,embed_dim]
        #  [2,3,224,224]->[2,56,56,96]
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # feats = []
        # for i, layer in enumerate(self.layers):
        #     if i != self.num_layers - 1:
        #         feats.append(x)
        #     x = layer(x)
        # feats.append(x)
        
        x1 = self.layers[0](x)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3)
        
        feats = [x, x1, x2]
        x1 = self.down2(self.down1(x1))
        x2 = self.down2(x2)
        sum = x1 + x2 + x3 + x4
        feats.append(sum)
        return feats


class SwinUnetDecoder(nn.Module):
    def __init__(self,
                 num_classes: int = 1000,
                 embed_dim: int = 96,
                 window_size: int = 7,
                 depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 ):

        super(SwinUnetDecoder, self).__init__()

        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path = drop_path_rate
        self.norm_layer = norm_layer

        self.first_patch_expanding = PatchExpanding(dim=embed_dim * 2 ** (len(depths) - 1), norm_layer=norm_layer)

        self.layers_up = self.build_layers_up()
        self.skip_connection_layers = self.skip_connection()
        self.norm_up = norm_layer(embed_dim)
        self.final_patch_expanding = FinalPatchExpanding(dim=embed_dim, norm_layer=norm_layer)
        self.head = nn.Conv2d(in_channels=embed_dim, out_channels=num_classes, kernel_size=(1, 1), bias=False)

    def skip_connection(self):
        skip_connection_layers = nn.ModuleList()
        for i in range(self.num_layers - 1):
            dim = self.embed_dim * 2 ** (self.num_layers - 2 - i)
            layer = nn.Linear(dim * 2, dim)
            skip_connection_layers.append(layer)
        return skip_connection_layers

    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 1):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer)
            layers_up.append(layer)
        return layers_up

    def forward(self, feats):
        """
        input=[b,c,h,w]
        feats=[b,h/4,h/4,C],[b,h/8,h/8,2C],[b,h/16,h/16,4C],[b,h/32,h/32,8C]

        例子 如下 c=96
        input=[2,3,224,224],
        feats=[2,56,56,96],[2,28,28,192],[2,14,14,384],[2,7,7,768]
        """
        x = feats[-1]
        x = self.first_patch_expanding(x)
        for i, layer in enumerate(self.layers_up):
            x = torch.cat([x, feats[len(feats) - i - 2]], -1)
            x = self.skip_connection_layers[i](x)
            x = layer(x)

        x = self.norm_up(x)
        x = self.final_patch_expanding(x)

        x = rearrange(x, 'B H W C -> B C H W')
        x = self.head(x)
        return x

class projection_conv(nn.Module):
    """
    A non-linear neck in DenseCL
    The non-linear neck, fc-relu-fc, conv-relu-conv
    """

    def __init__(self, in_dim, hid_dim=2048, out_dim=128, s=4):
        super(projection_conv, self).__init__()
        self.is_s = s
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hid_dim, out_dim))
        self.mlp_conv = nn.Sequential(nn.Conv2d(in_dim, hid_dim, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(hid_dim, out_dim, 1))
        if self.is_s:
            self.pool = nn.AdaptiveAvgPool2d((s, s))
        else:
            self.pool = None

    def forward(self, x):
        # Global feature vector
        x1 = self.avgpool(x)
        x1 = x1.reshape(x1.size(0), -1)
        x1 = self.mlp(x1)

        # dense feature map
        if self.is_s:
            x = self.pool(x)                        # [N, C, S, S]
        x2 = self.mlp_conv(x)
        x2 = x2.view(x2.size(0), x2.size(1), -1)    # [N, C, SxS]

        return x1, x2



class SwinUnet(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 96,
                 window_size: int = 7,
                 depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm: bool = True):

        super(SwinUnet, self).__init__()

        self.encoder = SwinUnetEncoder(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm
        )

        self.decoder = SwinUnetDecoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class SwinUnet_Plus(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim: int = 96,
                 window_size: int = 7,
                 depths: tuple = (2, 2, 6, 2),
                 num_heads: tuple = (3, 6, 12, 24),
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm: bool = True):

        super(SwinUnet_Plus, self).__init__()

        self.encoder = SwinUnetEncoder(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm
        )

        self.decoder = SwinUnetDecoder(
            num_classes=num_classes,
            embed_dim=embed_dim,
            window_size=window_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        self.dense_projection_high = projection_conv(embed_dim*8)
        self.dense_projection_head = projection_conv(num_classes, hid_dim=1024)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



    def val(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        return output

    def forward(self, x):
        feature = self.encoder(x)
        output = self.decoder(feature)
        high_feature = self.dense_projection_high(feature[-1].permute(0,3,1,2))
        head_feature = self.dense_projection_head(output)
        return output, high_feature, head_feature


def get_swinunet(
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 4):

    if img_size == 224:
        window_size = 7
    elif img_size == 256:
        window_size = 8
    else:
        raise NotImplementedError

    model = SwinUnet(
        patch_size=patch_size,
        in_chans=in_channels,
        num_classes=num_classes,
        embed_dim=96,
        window_size=window_size,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model

def get_swinunet_plus(
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 4):

    if img_size == 224:
        window_size = 7
    elif img_size == 256:
        window_size = 8
    else:
        raise NotImplementedError

    model = SwinUnet_Plus(
        patch_size=patch_size,
        in_chans=in_channels,
        num_classes=num_classes,
        embed_dim=96,
        window_size=window_size,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        mlp_ratio=4.,
        qkv_bias=True,
        drop_rate=0.1,
        attn_drop_rate=0.1,
        drop_path_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    return model



if __name__ == '__main__':
    # x = torch.randn(2, 3, 224, 224)
    # model = get_swinunet(img_size=224, in_channels=3)
    # y = model(x)
    # print(y.shape)

    x = torch.randn(2, 3, 224, 224)
    model = get_swinunet_plus(img_size=224, in_channels=3)
    y,y1,y2 = model(x)
    print(y.shape)

    # from thop import profile
    # model = get_swinunet(img_size=224, in_channels=3)
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input,))
    # print("flops:{:.3f}G".format(flops /1e9))
    # print("params:{:.3f}M".format(params /1e6))

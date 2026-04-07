import torch 
from einops import rearrange, repeat
import torch.nn as nn
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from timm.models.layers import DropPath
import torch.nn.functional as F
from natten import NeighborhoodAttention2D  as nat
from basicsr.archs.arch_util import to_2tuple, trunc_normal_
# import thop
# from thop import profile

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class Mlp(nn.Module):

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
   
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x
    
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)
        
class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    
def DMA(x, layer_num):
    B, D, H, W = x.shape
    L = H*W
    x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
    if layer_num % 2!= 0: 
        x_hwwh =  torch.flip(x_hwwh, dims=[-1])
    return x_hwwh

class SSMBlock(nn.Module):
    def __init__(self, dims,
        d_state=16,
        d_conv=3,
        expand=2):
        super().__init__()
        self.linear_layer= nn.Linear(in_features= dims , out_features= dims)
        self.linear_layer_parallel= nn.Linear(in_features= dims , out_features= dims)
        self.dw_conv = nn.Conv2d(in_channels= dims, out_channels= dims,kernel_size= 3,padding = 1, groups= dims)
        self.act = nn.SiLU()
        self.mamba_grp = MambaVisionMixer(d_model= dims, d_state=d_state, d_conv=d_conv, expand= expand)
        self.norm = nn.LayerNorm(dims)
        self.linear_after_multiply = nn.Linear(in_features= dims , out_features= dims)
    def forward(self, x, H, W): 
        B, L, D= x.shape
        shortcut = x
        x= self.linear_layer(x)
        x= x.view(B, H, W, D).permute(0,3,1,2)
        x= self.act(self.dw_conv(x))
        x = x.flatten(2).transpose(1, 2)
        x= self.norm(self.mamba_grp(x))
        shortcut= self.act(self.linear_layer_parallel(shortcut))
        return self.linear_after_multiply(shortcut*x)
     
class basic_block(nn.Module):
    def __init__(self, dims,
        downsample = 2, 
        d_state=16,
        d_conv=3,
        expand=2, 
        num_heads= 6,
        kernel_size= 3,
        stride= 1, 
        dilation = 1,
        drop_path = 0.,
        norm_layer= nn.LayerNorm):
        
        super().__init__()  
        self.norm_layer1= norm_layer(dims)
        self.norm_layer2= norm_layer(dims)
        self.nat1= nat(embed_dim= dims*num_heads, num_heads= num_heads, kernel_size= kernel_size, stride= stride,dilation = dilation)
        self.nat2= nat(embed_dim= dims*num_heads, num_heads= num_heads, kernel_size= kernel_size, stride= stride,dilation = dilation)
        self.mamba_grp_pixel = MambaVisionMixer(d_model= dims, d_state=d_state, d_conv=d_conv, expand= expand)
        self.mamba_grp_region= MambaVisionMixer(d_model= dims, d_state=d_state, d_conv=d_conv, expand= expand)
        self.conv1= nn.Conv2d(in_channels= dims*2, out_channels= dims,kernel_size= 3,padding = 1)
        self.conv2= nn.Conv2d(in_channels= dims, out_channels= 2,kernel_size= 3,padding = 1,)
        self.act = nn.GELU()
        self.softmax= nn.Softmax(dim = 1)
        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor= downsample)
        self.pixel_shuffle= nn.PixelShuffle(upscale_factor= downsample)
        self.conv_after_downsample= nn.Conv2d(in_channels = dims*downsample**2, out_channels= dims, kernel_size =1, bias = True)
        self.conv_after_upsample= nn.Conv2d(in_channels= dims, out_channels= dims, kernel_size =1, bias = True)
        self.conv_before_upsample= nn.Conv2d(in_channels= dims , out_channels = dims*downsample**2, kernel_size= 1, bias = True)
        self.downsample= downsample
        self.linear_before_nat1= nn.Linear(in_features= dims, out_features= dims*num_heads)
        self.linear_before_nat2= nn.Linear(in_features= dims, out_features= dims*num_heads)
        self.linear_after_nat1= nn.Linear(in_features= dims*num_heads, out_features= dims)
        self.linear_after_nat2= nn.Linear(in_features= dims*num_heads, out_features= dims)
        self.num_heads= num_heads
        self.linear= nn.Linear(in_features= dims, out_features= dims)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, H, W):
        B, L, D= x.shape
        x_pixel = self.norm_layer1(x)
        x_region= self.norm_layer2(x)
        x_region = x_region.view(B, H, W,D).permute(0, 3, 1, 2)
        x_region = self.conv_after_downsample(self.pixel_unshuffle(x_region))
        x_region = self.mamba_grp_region(x_region.flatten(2).transpose(1,2)) # B,L,D
        x_pixel = self.mamba_grp_pixel(x_pixel) # B,L,D
        x_pixel= self.linear_after_nat1(self.nat1(self.linear_before_nat1(x_pixel).view(B, H, W, D*self.num_heads).contiguous()).flatten(1,2))
        x_region= self.linear_after_nat2(self.nat2(self.linear_before_nat2(x_region).view(B, H//self.downsample, W//self.downsample, D*self.num_heads).contiguous()).flatten(1,2))
        x_region= x_region.view(B, H//self.downsample, W//self.downsample, D).permute(0, 3, 1, 2)
        x_region = self.conv_after_upsample(self.pixel_shuffle(self.conv_before_upsample(x_region)))  # B,C, H, W        
        x_pixel = x_pixel.view(B, H, W, D).permute(0, 3, 1, 2)
        x_fuse= torch.cat((x_pixel, x_region), dim= 1)
        x_fuse= self.softmax(self.conv2(self.act(self.conv1(x_fuse))))
        w1= x_fuse[:,0:1,:,:]
        w2= x_fuse[:,1:2,:,:]
        
        x_= w1*x_pixel+ w2*x_region
        x_out = self.linear(x_.permute(0, 2,3,1).flatten(1,2))
        return x_out+ self.drop_path(x)
class OCAB(nn.Module):
    # overlapping cross-attention block

    def __init__(self, dim,
                input_resolution,
                window_size,
                overlap_ratio,
                num_heads,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2,
                norm_layer=nn.LayerNorm
                ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3,  bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), stride=window_size, padding=(self.overlap_win_size-window_size)//2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim,dim)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
    
    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2) # 3, b, c, h, w
        q = qkv[0].permute(0, 2, 3, 1) # b, h, w, c
        kv = torch.cat((qkv[1], qkv[2]), dim=1) # b, 2*c, h, w

        # partition windows
        q_windows = window_partition(q, self.window_size)  # nw*b, window_size, window_size, c
        q_windows = q_windows.view(-1, self.window_size * self.window_size, c)  # nw*b, window_size*window_size, c

        kv_windows = self.unfold(kv) # b, c*w*w, nw
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch', nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous() # 2, nw*b, ow*ow, c
        k_windows, v_windows = kv_windows[0], kv_windows[1] # nw*b, ow*ow, c

        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, nq, d
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3) # nw*b, nH, n, d

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1)  # ws*ws, wse*wse, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, ws*ws, wse*wse
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = window_reverse(attn_windows, self.window_size, h, w)  # b h w c
        x = x.view(b, h * w, self.dim)

        x = self.proj(x) + shortcut

        x = x + self.mlp(self.norm2(x))
        return x
class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class main_block(nn.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 in_chans, 
                 dims,
                 kernel_size, 
                 dilation, 
                 num_heads,
                 num_blocks,
                 stride=1,
                 downsample=2, 
                 upscale=2,
                 drop_rate=0.2,
                 drop_path_rate=0.2,
                 window_size= 8,
                 overlap_ratio= 0.5, 
                 ape = False,
                 upsampler= '',
                 d_state=16,
                 d_conv=3,
                 expand=2, 
                 norm_layer= nn.LayerNorm, 
                 resi_connection='1conv', 
                 img_range= 1.
                 ):
        super().__init__()
        self.in_chans= in_chans
        self.dims = dims
        self.kernel_size= kernel_size
        self.dilation= dilation
        self.num_heads= num_heads
        self.stride= stride
        self.downsample = downsample
        self.d_state= d_state
        self.d_conv= d_conv
        self.expand= expand
        self.norm_layer= norm_layer
        self.upscale= upscale
        self.upsampler = upsampler
        num_feat = 64
        num_out_chans = in_chans
        self.window_size= window_size
        self.overlap_ratio = overlap_ratio
        self.patch_norm = norm_layer
        self.img_range= img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(in_chans, dims, 3, 1, 1)
        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_blocks = num_blocks
        self.ape = ape
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dims,
            embed_dim=dims,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=dims,
            embed_dim=dims,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers= nn.ModuleList()
        for i_layer in range(num_blocks):
            current_dpr = drop_path_rate * (i_layer / (num_blocks- 1))
            layer= basic_block(
                dims = dims, 
                downsample=downsample, 
                d_state=d_state, 
                d_conv=d_conv,
                expand = expand, 
                num_heads=num_heads, 
                kernel_size=kernel_size, 
                stride = stride, 
                dilation = dilation, 
                drop_path=current_dpr, 
                norm_layer=norm_layer
            )
            self.layers.append(layer)
        self.norm = norm_layer(dims)
        
        self.ocab = OCAB(dim = dims, input_resolution=(img_size, img_size), window_size=window_size, overlap_ratio= overlap_ratio, num_heads= num_heads)
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(dims, dims, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, dims))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        
        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(dims, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_chans, 3, 1, 1)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def calculate_rpi_oca(self):
        # calculate relative position index for OCA
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 2, ws*ws

        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 2, wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]   # 2, ws*ws, wse*wse

        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws, wse*wse, 2
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1

        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index
    
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
    
    def forward_features(self, x):
        x_size= (x.shape[2], x.shape[3])

        # Calculate attention mask and relative position index in advance to speed up inference. 
        # The original code is very time-consuming for large window size.
        rpi_oca= self.relative_position_index_OCA

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size[0], x_size[1])

        x = self.norm(x)  # b seq_len c
        x= self.ocab(x, x_size, rpi_oca)
        x = self.patch_unembed(x, x_size)
        return x
    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean

        return x
# dims = 3
# H, W= 64, 64
# device = 'cuda'
# input_tensor= torch.rand(1, dims,H, W).cuda()

# model = main_block(
#     img_size= 64,
#     patch_size=1,
#     in_chans=3,
#     dims=144, 
#     kernel_size= 3, 
#     dilation=1, 
#     num_heads=6, 
#     num_blocks= 16, 
#     stride=1,
#     downsample=2, 
#     upscale=2,
#     drop_rate=0.1, 
#     drop_path_rate=0.2, 
#     window_size= 8, 
#     overlap_ratio= 0.5,
#     upsampler='pixelshuffle'
#     ).to(device)
# model.eval()

# # ---------------------------------------------------------
# # 2. Measure FLOPs and Parameters (Computational Complexity)
# # ---------------------------------------------------------
# # Note: thop expects (batch, channel, height, width) usually, 
# # but your forward accepts (x, H, W). We use custom_ops if needed.
# # For simplicity, we wrap it to pass H and W implicitly or just pass args.
# try:
#     flops, params = profile(model, inputs=(input_tensor,), verbose=False)
#     print(f"--- Complexity ---")
#     print(f"Params: {params / 1e6:.2f} M")       # Matches 'Params(M)' in paper
#     print(f"FLOPs:  {flops / 1e9:.2f} G")        # Matches 'FLOPS(G)' in paper
# except Exception as e:
#     print(f"FLOPs calculation failed (common with custom kernels): {e}")

# # ---------------------------------------------------------
# # 3. Measure Inference Time (Latency)
# # ---------------------------------------------------------
# # Warmup to initialize CUDA context (crucial for accuracy)
# for _ in range(50):
#     _ = model(input_tensor)

# # Setup CUDA events for precise timing
# start_event = torch.cuda.Event(enable_timing=True)
# end_event = torch.cuda.Event(enable_timing=True)

# iterations = 100
# torch.cuda.synchronize() # Wait for everything to finish
# start_event.record()

# with torch.no_grad():
#     for _ in range(iterations):
#         _ = model(input_tensor)

# end_event.record()
# torch.cuda.synchronize() # Wait for GPU to finish

# elapsed_time_ms = start_event.elapsed_time(end_event) / iterations
# print(f"\n--- Timing ---")
# print(f"Avg Inference Time: {elapsed_time_ms:.2f} ms") # Matches 'Time(ms)' in paper

# # ---------------------------------------------------------
# # 4. Measure Peak Memory (Space)
# # ---------------------------------------------------------
# torch.cuda.reset_peak_memory_stats()
# torch.cuda.empty_cache()

# with torch.no_grad():
#     _ = model(input_tensor)

# max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2) # Convert Bytes to MB
# print(f"\n--- Memory ---")
# print(f"Peak Memory: {max_memory:.2f} MB")       # Matches 'Memory(MB)' in paper
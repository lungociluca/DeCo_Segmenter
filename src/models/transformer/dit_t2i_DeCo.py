import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import einops
import math

import torchvision
from functools import lru_cache
from src.models.layers.attention_op import attention
from src.models.layers.rope import apply_rotary_emb, precompute_freqs_cis_ex2d as precompute_freqs_cis_2d
from src.models.layers.time_embed import TimestepEmbedder as TimestepEmbedder
from src.models.layers.patch_embed import Embed as Embed
from src.models.layers.swiglu import SwiGLU as FeedForward
from src.models.layers.rmsnorm import RMSNorm as Norm

import config as local_config

def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def bulid_cross_attention_tuples(q, k, v, ky, vy):
    cross_attn_type: local_config.CrossAttnType
    cross_attn_tuples = []
    for cross_attn_type in local_config.cross_attention_types:
        if cross_attn_type == local_config.CrossAttnType.Q_K:
            q_key = q
            k_key = ky
        elif cross_attn_type == local_config.CrossAttnType.K_K:
            q_key = k
            k_key = ky
        elif cross_attn_type == local_config.CrossAttnType.V_V:
            q_key = v
            k_key = vy
        else:
            raise NotImplemented("Cannot build tuples for computing cross attention")
        cross_attn_tuples.append(
            (cross_attn_type.value, q_key, k_key)
        )
    return cross_attn_tuples

def set_attention_aggregation():
    config_aggregation_method: local_config.AttentionAggregateMethod = local_config.attention_aggregate_method
    if config_aggregation_method == local_config.AttentionAggregateMethod.NO_OP:
        return Attention.refine_no_operation
    elif config_aggregation_method == local_config.AttentionAggregateMethod.MERGE_WITH_SELF_ATTN:
        return Attention.refine_via_multiplication
    else:
        raise NotImplemented()
    
def set_head_aggregation():
    config_head_agg_method: local_config.HeadAggregateMethod = local_config.head_aggregate_method
    if config_head_agg_method == local_config.HeadAggregateMethod.MEAN:
        return Attention.merge_head_average
    elif config_head_agg_method == local_config.HeadAggregateMethod.WEIGHTED_MEAN:
        return Attention.merge_heads_weighted_mean
    else:
        raise NotImplemented()
    
def set_projection_matrix_computation():
    config_projection_method: local_config.AttentionMapsProjection = local_config.attention_map_projection
    if config_projection_method == local_config.AttentionMapsProjection.NO_OP:
        return Attention.projection_no_operation
    elif config_projection_method == local_config.AttentionMapsProjection.ORTHOGONAL_TO_PRINCIPAL_COMP:
        return Attention.principal_comp_projection
    else:
        raise NotImplemented()
    
def set_maps_weighting():
    config_weighting_method: local_config.MapsWeighting = local_config.maps_weighting
    if config_weighting_method == local_config.MapsWeighting.NO_OP:
        return Attention.maps_weighting_no_op
    elif config_weighting_method == local_config.MapsWeighting.POOLING_OVER_COS_SIMILARITY:
        return Attention.maps_weighting_pooling
    else:
        raise NotImplemented()

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_x = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.kv_y = nn.Linear(dim, dim*2, bias=qkv_bias)

        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.aggregate_attn_maps = set_attention_aggregation()
        self.aggregate_heads = set_head_aggregation()
        self.get_projection = set_projection_matrix_computation()
        self.maps_weighting = set_maps_weighting()

        self.default_prompt_emb = None

        # self.unbias_matrix = self.remove_bias()
    
    def set_default_text_emb(self, y):
        y = y[:, [4], :]
        kv_y = self.kv_y(y).reshape(1, -1, 2, self.num_heads, 1536 // self.num_heads).permute(2, 0, 3, 1, 4)
        ky = kv_y[0]
        # ky = self.k_norm(ky.contiguous())
        self.default_prompt_emb = ky

    @staticmethod
    def softmax_for_each_prompt(attn_map, no_prompts):
        """
        attn_maps: [b, h, img_patches, img_patches + no_prompts * tokens]
        """
        cross_attn_maps_list = []
        image_patches = attn_map.shape[2]
        text_patches = 128
        self_attn_maps = torch.softmax(
            attn_map[:, :, :, 0:image_patches] / local_config.self_attn_softmax_temperature, 
            dim=-1
        ).contiguous()
        
        for i in range(no_prompts):
            cross_attn_maps_list.append(
                attn_map[:, :, :, 0 + i * text_patches + local_config.idx_token_of_interest], 
            )
        return self_attn_maps[0], torch.stack(cross_attn_maps_list, dim=-1)[0].contiguous()


    @staticmethod
    def merge_heads_weighted_mean(attn_maps):
        # head_mean = torch.zeros(cross_attn[:,0,:].shape).to(local_config.device)
        # head_weight = torch.zeros(cross_attn.shape[1]).to(local_config.device)
        # for ii in range(cross_attn.shape[1]):
        #     head_weight[ii] += cross_attn[:, ii, :].sum() / attn[:, ii, :, :].sum()
        # head_weight = torch.softmax(head_weight, dim=0)

        # head_mean = head_weight.unsqueeze(0).unsqueeze(2) * cross_attn
        # return head_mean.mean(1)
        return None
    
    @staticmethod
    def merge_head_average(attn_maps):
        return attn_maps.mean(0)
    
    @staticmethod
    def refine_via_multiplication(cross_attn, self_attn):
        """
        self: (img_patches, img_patches)
        cross: (img_patches, no_prompts)
        """
        aggregated = torch.matmul(
            self_attn.unsqueeze(0).repeat(cross_attn.shape[-1], 1, 1), 
            einops.rearrange(cross_attn, "p t -> t p").unsqueeze(-1)
        )
        return einops.rearrange(aggregated.sum(-1), "t p -> p t")

    @staticmethod
    def refine_no_operation(cross_attn, self_attn):
        return cross_attn
    
    @staticmethod
    def projection_no_operation(uncond_maps, img_h, img_w):
        return None
    
    @staticmethod
    def principal_comp_projection(uncond_maps, img_h, img_w):
        B, H, P, D= uncond_maps.shape
        uncond_maps = uncond_maps.reshape(H, img_h, img_w)[1].unsqueeze(0)
        uncond_maps = einops.rearrange(uncond_maps, 'c h w -> c (h w)')
        basis = torch.linalg.svd(
            uncond_maps,
            full_matrices=False
        )[0]
        basis = basis[:, :local_config.unbiasing_components_count].contiguous()
        basis_mtrx = torch.eye(H, dtype=uncond_maps.dtype, device=local_config.device) - basis @ basis.T
        return basis_mtrx
    
    @staticmethod
    def remove_bias():
        texture_features = torch.load('texture_qs.pth').to(local_config.device)
        B, H, P, D = texture_features.shape
        texture_features = einops.rearrange(texture_features, 'b h p d -> (h d) (b p)')
        print("text features", texture_features.shape)
        texture_features = texture_features - texture_features.mean()
        basis = torch.linalg.svd(
            texture_features,
            full_matrices=False
        )[0]
        print('basis before', basis.shape)
        basis = basis[:, :local_config.unbiasing_components_count].contiguous()
        print('basis', basis.shape)
        basis_mtrx = torch.eye(H * D, dtype=texture_features.dtype, device=local_config.device) - basis @ basis.T
        print('matrix', basis_mtrx.shape)
        return basis_mtrx.to("cpu")
    
    @staticmethod
    def apply_projection(projection, cross_attn):
        """
        projection: (B, H, IMG_P, IMG_P) or (B, H, IMG_P)
        cross_attn: (B, H, IMG_P)
        """
        if projection is not None:
            return cross_attn - projection
        else:
            return cross_attn
        
    @staticmethod
    def maps_weighting_no_op(cross_attn, kx, ky):
        return torch.eye(cross_attn.shape[-1]).to(local_config.device)
    
    @staticmethod
    def maps_weighting_pooling(cross_attn, kx, ky):
        print(f"cross {cross_attn.shape}, kx {kx.shape}, ky {ky.shape}")
        _, patches, prompts_count = cross_attn.shape
        heads = kx.shape[1]
        tokens_count = ky.shape[2] // prompts_count
        
        interest_tokens_k = torch.stack(
            [ky[0, :, i * tokens_count + local_config.idx_token_of_interest, :] for i in range(prompts_count)],
            dim=0
        )
        aggregated_kx = torch.matmul(cross_attn[0].unsqueeze(-1), einops.rearrange(kx[0], 'h p d -> p (h d)').unsqueeze(-2))
        aggregated_kx = einops.rearrange(aggregated_kx.mean(0), 't (h d) -> t h d', h=heads)

        cos = torch.nn.CosineSimilarity(dim=2)
        output = cos(interest_tokens_k, aggregated_kx)
        output = output.mean(-1)
        output = output / torch.max(output)
        return torch.diagflat(output).to(local_config.device)
    
    def forward_orig(self, x: torch.Tensor, y, pos) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q.contiguous())
        kx = self.k_norm(kx.contiguous())
        # q, kx = apply_rotary_emb(q, kx, freqs_cis=pos)
        kv_y = self.kv_y(y).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky.contiguous())

        k = torch.cat([kx, ky], dim=2)
        v = torch.cat([vx, vy], dim=2)

        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()

        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_cosine(self, x: torch.Tensor, y, pos, extra_dict=None, attention_maps_dir: str = None, img_h: int = None, 
                img_w: int = None, eval_mode=False):
        B, N, C = x.shape
        x_noise = torchvision.transforms.functional.normalize(
            torch.zeros(x.shape),
            mean=x.mean(), std=x.std(),
        ).to(local_config.device)
        no_prompts = y.shape[0]
        # IMAGE PROJECTIONS
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv_x_noise = self.qkv_x(x_noise).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        q, kx, vx = qkv_x[0], qkv_x[1], qkv_x[2]
        kx_noise = qkv_x_noise[1]
        
        q = self.q_norm(q.contiguous())
        kx = self.k_norm(kx.contiguous())
        
        # PROMPT PROJECTIONS
        kv_y = self.kv_y(y).reshape(no_prompts, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky, vy = kv_y[0], kv_y[1]
        ky = self.k_norm(ky.contiguous())

        # kx, debias_mtrx = Attention.remove_bias(kx, kx_noise)

        ky_tokens = torch.stack(
            [ky[i, :, local_config.idx_token_of_interest, :].unsqueeze(0) for i in range(no_prompts)],
            dim=-2
        )

        ky_tokens, _ = Attention.remove_bias(ky_tokens, ky_tokens[:,:,-1,:].unsqueeze(2))
        cos = torch.nn.CosineSimilarity(dim=-1)
        kx = kx.unsqueeze(3)
        ky_tokens = ky_tokens.unsqueeze(2)

        aggregated_attn_maps = cos(kx, ky_tokens)
        aggregated_attn_maps = aggregated_attn_maps.mean(1)

        aggregated_attn_maps = einops.rearrange(aggregated_attn_maps, "b p c -> b (p c)")
        aggregated_attn_maps = aggregated_attn_maps / aggregated_attn_maps.max()
        aggregated_attn_maps = einops.rearrange(aggregated_attn_maps, " b (p c) -> b p c", p=N)

        if attention_maps_dir is not None:
            for i in range(no_prompts):
                aggregated_slice = aggregated_attn_maps[:, :, i]
                self._save_attention_maps_as_images(torch.ones((1, 2120, 2120)), aggregated_slice, aggregated_slice, aggregated_slice, attention_maps_dir, i, "", img_h, img_w,
                                                    extra_dict=extra_dict)

        return torch.zeros(x.shape).to(local_config.device), aggregated_attn_maps
    
    def forward_attention(self, x: torch.Tensor, y, pos, token_lengths, extra_dict=None, attention_maps_dir: str = None, img_h: int = None, 
                img_w: int = None, eval_mode=False,l=1000):
        B, N, C = x.shape
        no_prompts = y.shape[0]
        # IMAGE PROJECTIONS
        qkv_x = self.qkv_x(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = qkv_x[0]
        q = self.q_norm(q.contiguous())
        
        # print(q.dtype, self.unbias_matrix.dtype)
        # q = torch.matmul(self.unbias_matrix, einops.rearrange(q, 'b h p d -> b p (h d)').unsqueeze(-1).to('cpu')).squeeze(-1).to(local_config.device)
        # q = einops.rearrange(q, 'b p (h d) -> b h p d', h=self.num_heads)

        # PROMPT PROJECTIONS
        y = torch.cat([y[[i], 4:token_idx+1, :].mean(1) for i, token_idx in enumerate(token_lengths)], dim=0)
        kv_y = self.kv_y(y).reshape(no_prompts, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        ky = kv_y[0]
        # ky = self.k_norm(ky.contiguous())
        ky = torch.cat([ky, self.default_prompt_emb.repeat(no_prompts,1,1,1)], dim=2)
        # ky = torch.clamp(ky, min=-200, max=200)

        scale = 1 / math.sqrt(q.size(-1))
        cross_attn_maps = q @ ky.transpose(-2, -1) * scale
        th = torch.nn.Threshold(0.0, 0.0)
        min_max = lambda x,ii: (x[:,:,:,ii] - x[:,:,:,ii].min()) / (x[:,:,:,ii].max() - x[:,:,:,ii].min())
        for ii in range(2):
            print(ii, cross_attn_maps[:,:,:,ii].min(), cross_attn_maps[:,:,:,ii].max())
            cross_attn_maps[:,:,:,ii] = min_max(cross_attn_maps, ii)
        sh = lambda map: map#torch.softmax(map // 2000, dim=2)

        cross_attn_maps = th(sh(cross_attn_maps[:, :, :, 0]) - sh(cross_attn_maps[:, :, :, 1]))
        aggregated_attn_maps = cross_attn_maps.mean(1)
        
        for i in range(no_prompts):
            aggregated_attn_maps[i] = (aggregated_attn_maps[i] - aggregated_attn_maps[i].min()) / (aggregated_attn_maps[i].max() - aggregated_attn_maps[i].min())
        aggregated_attn_maps = einops.rearrange(aggregated_attn_maps, 'b p-> p b').unsqueeze(0)
        
        # for t in range(15):
        #     for i in range(no_prompts):
        #         aggregated_slice = aggregated_attn_maps[:, :, i, t]
        #         self._save_attention_maps_as_images(aggregated_slice, aggregated_slice, aggregated_slice, aggregated_slice, attention_maps_dir, i, "", img_h, img_w,
        #                                             extra_dict=extra_dict, idx=t,l=l)
        
        if attention_maps_dir is not None:
            for i in range(no_prompts):
                aggregated_slice = aggregated_attn_maps[:, :, i]
                self._save_attention_maps_as_images(aggregated_slice, aggregated_slice, aggregated_slice, aggregated_slice, attention_maps_dir, i, "", img_h, img_w,
                                                    extra_dict=extra_dict,l=l)                

        return self.forward_orig(x, y, pos) if local_config.dit_blocks > 1 else torch.zeros(x.shape, device=x.device), aggregated_attn_maps
    
    def forward(self, x: torch.Tensor, y, pos, token_lengths, extra_dict=None, attention_maps_dir: str = None, img_h: int = None, 
            img_w: int = None, eval_mode=False,l=1000):
        config_forward_method: local_config.ForwardMethod = local_config.forward_method
        if config_forward_method == local_config.ForwardMethod.ATTENTION:
            return self.forward_attention(x, y, pos, token_lengths, extra_dict, attention_maps_dir, img_h, img_w, eval_mode,l=l)
        elif config_forward_method == local_config.ForwardMethod.COSINE_SIMILARITY:
            return self.forward_cosine(x, y, pos, extra_dict, attention_maps_dir, img_h, img_w, eval_mode)
        else:
            raise NotImplemented()
    
    @staticmethod
    def _save_attention_maps_as_images(self_attn_maps, cross_attn_maps, mixed_attention_maps, uncond_attn_maps, save_dir, slice_idx, 
                                       label, img_h=None, img_w=None, extra_dict=None, idx=999,l=1000):
        """
        Save attention maps as images to specified directory.
        
        Args:
            self_attn_maps: Self-attention maps, shape (B, N, N)
            cross_attn_maps: Cross-attention maps, shape (B, N, M)
            save_dir: Directory to save the images
            img_h: Image height (in patches)
            img_w: Image width (in patches)
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Determine spatial dimensions
        N = self_attn_maps.shape[1]
        if img_h is None or img_w is None:
            # Assume square spatial arrangement
            spatial_size = int(N ** 0.5)
            img_h, img_w = spatial_size, spatial_size
        
        # Save cross-attention maps
        cross_attn_np = cross_attn_maps.cpu().detach().numpy()
        for b in range(cross_attn_np.shape[0]):
            # shape (N, M) - reshape query dimension to spatial
            attn_map = cross_attn_np[b]  # shape (N, M)
            attn_spatial = attn_map.reshape(img_h, img_w, -1)
            # Average over text tokens (last dimension)
            attn_spatial_avg = attn_spatial.mean(axis=2)
            
            fig, ax = plt.subplots(figsize=(img_w, img_h))
            im = ax.imshow(attn_spatial_avg, cmap='viridis', aspect='equal')
            ax.set_title(f'Cross-Attention - Batch {b}')
            ax.set_xlabel('Image Width')
            ax.set_ylabel('Image Height')
            plt.colorbar(im, ax=ax)
            
            save_path = os.path.join(save_dir, f'cross_attn_img{extra_dict["img_id"]}_{label}_lyr_{l}_{b}_{idx}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)


class FlattenDiTBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def set_default_text_emb(self, emb):
        self.attn.set_default_text_emb(emb)

    def forward(self, x, y, c, pos, token_lengths, extra_dict=None, attn_maps_dir=None, img_h=None, img_w=None,l=1000):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x, attn_maps = self.attn(modulate(self.norm1(x), shift_msa, scale_msa), y, pos, token_lengths, extra_dict, attn_maps_dir, img_h, img_w, eval_mode=local_config.eval,l=l)
        x = x + gate_msa * x
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x, attn_maps

class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs):
        super().__init__()
        self.max_freqs = max_freqs
        self.hidden_size_input = hidden_size_input
        self.embedder = nn.Sequential(
            nn.Linear(in_channels+max_freqs**2, hidden_size_input, bias=True),
        )

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        pos = precompute_freqs_cis_2d(self.max_freqs ** 2 * 2, patch_size, patch_size)
        pos = pos[None, :, :].to(device=device, dtype=dtype)
        return pos


    def forward(self, inputs):
        B, P2, C = inputs.shape
        patch_size = int(P2 ** 0.5)
        device = inputs.device
        dtype = inputs.dtype
        dct = self.fetch_pos(patch_size, device, dtype)
        dct = dct.repeat(B, 1, 1)
        inputs = torch.cat([inputs, dct], dim=-1)
        inputs = self.embedder(inputs)
        return inputs

class NerfBlock(nn.Module):
    def __init__(self, hidden_size_s, hidden_size_x, mlp_ratio=4):
        super().__init__()
        self.param_generator1 = nn.Sequential(
            nn.Linear(hidden_size_s, 2*hidden_size_x**2*mlp_ratio, bias=True),
        )
        self.norm = Norm(hidden_size_x, eps=1e-6)
        self.mlp_ratio = mlp_ratio
    def forward(self, x, s):
        batch_size, num_x, hidden_size_x = x.shape
        mlp_params1 = self.param_generator1(s)
        fc1_param1, fc2_param1 = mlp_params1.chunk(2, dim=-1)
        fc1_param1 = fc1_param1.view(batch_size, hidden_size_x, hidden_size_x*self.mlp_ratio)
        fc2_param1 = fc2_param1.view(batch_size, hidden_size_x*self.mlp_ratio, hidden_size_x)

        # normalize fc1
        normalized_fc1_param1 = torch.nn.functional.normalize(fc1_param1, dim=-2)
        # mlp 1
        res_x = x
        x = self.norm(x)
        x = torch.bmm(x, normalized_fc1_param1)
        x = torch.nn.functional.silu(x)
        x = torch.bmm(x, fc2_param1)
        x = x + res_x
        return x

class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
    def forward(self, x):
        x = self.linear(x)
        return x

class TextRefineAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = Norm(self.head_dim)
        self.k_norm = Norm(self.head_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv_x = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv_x[0], qkv_x[1], qkv_x[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.view(B, self.num_heads, -1, C // self.num_heads)  # B, H, N, Hc
        k = k.view(B, self.num_heads, -1, C // self.num_heads).contiguous()  # B, H, N, Hc
        v = v.view(B, self.num_heads, -1, C // self.num_heads).contiguous()
        x = attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TextRefineBlock(nn.Module):
    def __init__(self, hidden_size, groups,  mlp_ratio=4, ):
        super().__init__()
        self.norm1 = Norm(hidden_size, eps=1e-6)
        self.attn = TextRefineAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm2 = Norm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)

        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        patch_size,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        self.patch_size = patch_size

        self.cond_embed = nn.Linear(z_channels, patch_size**2*model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)
        
        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        c = self.cond_embed(c)

        y = c.reshape(c.shape[0], self.patch_size**2, -1)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x)

class PixNerDiT(nn.Module):
    def __init__(
            self,
            in_channels=4,
            num_groups=12,
            hidden_size=1152,
            decoder_hidden_size=64,
            num_encoder_blocks=18,
            num_decoder_blocks=4,
            num_text_blocks=4,
            patch_size=2,
            txt_embed_dim=1024,
            txt_max_length=100,
            weight_path=None,
            load_ema=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.decoder_hidden_size = decoder_hidden_size
        self.num_encoder_blocks = num_encoder_blocks
        self.num_decoder_blocks = num_decoder_blocks
        self.num_blocks = self.num_encoder_blocks + self.num_decoder_blocks
        self.num_text_blocks = num_text_blocks
        self.patch_size = patch_size
        self.txt_embed_dim = txt_embed_dim
        self.txt_max_length = txt_max_length
        self.s_embedder = Embed(in_channels*patch_size**2, hidden_size, bias=True)
        self.x_embedder = NerfEmbedder(in_channels, decoder_hidden_size, max_freqs=8)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = Embed(txt_embed_dim, hidden_size, bias=True, norm_layer=Norm)
        self.y_pos_embedding = torch.nn.Parameter(
            torch.randn(1, txt_max_length, hidden_size),
            requires_grad=True
        )

        self.blocks = nn.ModuleList([
            FlattenDiTBlock(self.hidden_size, self.num_groups) for _ in range(local_config.dit_blocks)
        ])
        
        # self.dec_net = SimpleMLPAdaLN(
        #     in_channels=self.decoder_hidden_size,
        #     model_channels=self.decoder_hidden_size,
        #     out_channels=self.in_channels,  # for vlb loss
        #     z_channels=self.hidden_size,
        #     num_res_blocks=self.num_decoder_blocks,
        #     patch_size=self.patch_size,
        #     grad_checkpointing=False
        # )

        self.text_refine_blocks = nn.ModuleList([
            TextRefineBlock(self.hidden_size, self.num_groups) for _ in range(self.num_text_blocks)
        ])
        self.initialize_weights()
        self.precompute_pos = dict()
        self.weight_path = weight_path
        self.load_ema = load_ema

    def fetch_pos(self, height, width, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        else:
            pos = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
            self.precompute_pos[(height, width)] = pos
            return pos

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def set_default_prompt_emb(self, y):
        t = torch.zeros((1)).to(local_config.device) + 0.01 # TODO
        ypos = self.y_pos_embedding
        t = self.t_embedder(t.view(-1)).view(1, -1, self.hidden_size)
        y = self.y_embedder(y).view(1, -1, self.hidden_size) + ypos.to(y.dtype)

        condition = nn.functional.silu(t)
        for i, block in enumerate(self.text_refine_blocks):
            y = block(y, condition)

        for i in range(local_config.dit_blocks):
            self.blocks[i].set_default_text_emb(y)

    def forward(self, x, t, y, token_lengths, extra_dict=None):
        B, _, H, W = x.shape
        eval_mode = extra_dict["eval_mode"]
        prompts_count = y.shape[0]
        # y = y[prompts_count:]
        x = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        xpos = self.fetch_pos(H // self.patch_size, W // self.patch_size, x.device)
        ypos = self.y_pos_embedding
        t = self.t_embedder(t.view(-1)).view(B, -1, self.hidden_size)
        y = self.y_embedder(y).view(prompts_count, -1, self.hidden_size) + ypos.to(y.dtype)

        condition = nn.functional.silu(t)
        for i, block in enumerate(self.text_refine_blocks):
            y = block(y, condition)

        s = self.s_embedder(x)
        attention_maps_dir_format = os.path.join(local_config.attention_maps_dir, "{idx}_attn_maps")
        maps_array = []
        for i in range(self.num_encoder_blocks):
            s, maps = self.blocks[i](s, y, condition, xpos, token_lengths, extra_dict=extra_dict, attn_maps_dir=attention_maps_dir_format.format(idx=i), 
                               img_h=H // self.patch_size, img_w=W // self.patch_size, l=i)
            if eval_mode:
                maps_array.append(maps)
                # TODO
                if i == local_config.dit_blocks - 1:
                    maps = torch.stack(maps_array)[-1]
                    for i in range(maps.shape[-1]):
                        maps[:,:,i] = (maps[:, :, i] - maps[:, :, i].min()) / (maps[:, :, i].max() - maps[:, :, i].min())
                    # for pid in range(prompts_count):
                    #     Attention._save_attention_maps_as_images(maps[:,:,pid], maps, maps, maps, attention_maps_dir_format.format(idx=99), pid, "", H // self.patch_size,  img_w=W // self.patch_size,
                    #                                     extra_dict=extra_dict,l=99999990)  
                    return einops.rearrange(maps[0], "p b -> b p").reshape(prompts_count, H//self.patch_size, W//self.patch_size)
                
        # s = torch.nn.functional.silu(t + s)
        # batch_size, length, _ = s.shape
        # x = x.reshape(batch_size * length, self.in_channels, self.patch_size ** 2 )
        # x = x.transpose(1, 2)
        # s = s.view(batch_size * length, self.hidden_size)
        # x = self.x_embedder(x)

        # x = self.dec_net(x, s)
        
        # x = x.transpose(1, 2)
        # x = x.reshape(batch_size, length, -1)
        # x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(),
        #                              (H, W),
        #                              kernel_size=self.patch_size,
        #                              stride=self.patch_size)
        # return x.repeat(B, 1, 1, 1)
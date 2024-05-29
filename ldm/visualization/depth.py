import torch
from jaxtyping import Float, Bool
from torch import Tensor
from einops import repeat


def filter_depth(depth_map: Float[Tensor, "*batch height width"]) -> tuple[Float[Tensor, "*batch height width"], Bool[Tensor, "*batch height width"]]:
    """Filter infinite depth values based median depth value and return filter mask.
    """
    H, W = depth_map.shape[-2:]
    depth_map_filtered = depth_map.clone().detach()

    z_min = depth_map_filtered.flatten(start_dim=-2).min(dim=-1, keepdims=True).values[...,None]
    median_depth = torch.quantile(depth_map_filtered.flatten(start_dim=-2), 0.5, dim=-1, keepdims=True)[...,None]

    # Filter and replace outliers by filtered max.
    valid_mask = (depth_map_filtered < median_depth * 1_000_000)
    full_min = repeat(z_min, '... () () -> ... h w', h=H, w=W)
    filtered_max =  torch.where(valid_mask, depth_map_filtered, full_min).flatten(start_dim=-2).max(dim=-1, keepdims=True).values[...,None]
    full_filtered_max = repeat(filtered_max, '... () () -> ... h w', h=H, w=W)
    depth_map_filtered = torch.where(valid_mask, depth_map_filtered, full_filtered_max * 1.5)
    
    return depth_map_filtered, valid_mask
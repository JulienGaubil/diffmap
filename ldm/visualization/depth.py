import torch
from jaxtyping import Float, Bool
from torch import Tensor


def filter_depth(depth_map: Float[Tensor, "height width"]) ->  tuple[Float[Tensor, "height width"], Bool[Tensor, "height width"]]:
    """Filter infinite depth values based on an heuristic and return filter mask.
    """
    depth_map_filtered = depth_map.clone().detach()

    z_max = depth_map_filtered.max().item()
    median_depth = torch.quantile(depth_map_filtered, 0.5).item()

    if z_max > 1_000_000 * median_depth: #heuristics to detect when infinite depth value is used
        filtered_max = depth_map[depth_map < z_max].max()
        filter_mask = depth_map_filtered == z_max
        depth_map_filtered[filter_mask] =  filtered_max * 1.5
    
    else:
        filter_mask = torch.zeros_like(depth_map, dtype=torch.bool)
    
    return depth_map_filtered, filter_mask
    
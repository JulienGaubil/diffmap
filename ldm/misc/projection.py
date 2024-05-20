import torch
import torch.nn.functional as F

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..data.utils.camera import Camera, Intrinsics, Extrinsics, pixel_grid_coordinates
from ldm.modules.flowmap.model.projection import sample_image_grid

def compute_flow_projection(
        src_depth_map: Float[Tensor, "height width"],
        src_camera: Camera,
        trgt_camera: Camera,
) -> Float[Tensor, "height width vu=2"]:
    """Compute flow by un-projecting and reprojecting depth in next frame. Values in range ]-1,1[ for image coordinates in [0,1].
    """
    H_trgt, W_trgt = src_depth_map.size()
    HW = pixel_grid_coordinates(H_trgt, W_trgt)
    pixel_coordinates = rearrange(HW, 'h w c -> c (h w)')
    depths = rearrange(src_depth_map, 'h w -> 1 (h w)')

    # Un-project depths to world and project back in next frame.
    world_pts = src_camera.pixel_to_world(pixel_coordinates, depths=depths)
    pixel_coordinates_warped, _ = trgt_camera.world_to_pixel(world_pts)

    # Compute flows.
    HW_warped = rearrange(pixel_coordinates_warped, 'c (h w) -> h w c', h=H_trgt, w=W_trgt)
    flow = (HW_warped - HW).to(torch.float32)

    # Normalize flow.
    hw = torch.tensor((H_trgt, W_trgt), dtype=torch.float32)
    flow = flow / hw[None,None,:]

    # Swap to flow indexing uv -> vu to match flowmap
    flow = flow[:,:,[1,0]]

    return flow

def compute_consistency_mask(
    src_frame: Float[Tensor, "3 height width"],
    trgt_frame: Float[Tensor, "3 height width"],
    flow: Float[Tensor, "height width vu=2"],
) -> Float[Tensor, "height width"]:
    
    # Manual warping
    _, H, W = src_frame.shape
    b = 1
    f = 2
    flow = flow[None,None,:,:,:]
    src_frames = src_frame[None,:,:,:]
    trgt_frames = trgt_frame[None,:,:,:]
    source_xy, _ = sample_image_grid((H, W))
    target_xy = source_xy + rearrange(flow, "b f h w xy -> (b f) h w xy")

    # Gets target pixel for every source pixel warped with flow.
    target_pixels = F.grid_sample(
        trgt_frames,
        target_xy * 2 - 1,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ) #((bf) c h w)

    # Map pixel color differences to mask weights.
    deltas = (src_frames - target_pixels).abs().max(dim=1).values #((bf) h w)
    mask = rearrange((1 - deltas) ** 8, "(b f) h w -> b f h w", b=b, f=f - 1)
    return mask.squeeze()
import torch
import torch.nn.functional as F

from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..data.utils.camera import Camera, Intrinsics, Extrinsics, pixel_grid_coordinates
from ldm.thirdp.flowmap.flowmap.model.projection import sample_image_grid

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

def intrinsics_from_3D(
    surfaces: Float[Tensor, "batch frame height width 3"],
    min_focal=0.,
    max_focal=torch.inf,
    epsilon = 1e-5,
    infinity: float = 1e8,
) -> Float[Tensor, "batch frame 3 3"]:
    """Estimate normalized camera focal lens from 2D-3D correspondences in camera coordinate space using Weiszfeld's algorithm.
    Assumes optical center at the center of the image, returns normalized intrinsics.
    Adapted from https://github.com/naver/dust3r/blob/main/dust3r/post_process.py
    """
    B, F, H, W, THREE = surfaces.shape
    assert THREE == 3
    pts3d = rearrange(surfaces, 'b f h w c -> (b f) h w c')
    device=pts3d.device

    # Centered pixel grid.
    pp = torch.tensor((0.5,0.5), device=device)
    xy, _ = sample_image_grid((H,W), device=device)
    pixels = xy.view(1, -1, 2) - pp.view(-1, 1, 2)  # (batch, (height width), xy=2)
    pts3d = rearrange(pts3d, 'bf h w c -> bf (h w) c') # (batch, (height width), xyz=3)

    # Initialization: non-weighted least square closed-form.
    xy_over_z = pts3d[..., :2] / (pts3d[..., -1:] + epsilon)
    xy_over_z = xy_over_z.nan_to_num(posinf=infinity, neginf=-infinity)
    dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
    dot_xy_xy = xy_over_z.square().sum(dim=-1)
    focal = dot_xy_px.sum(dim=1) / (dot_xy_xy.sum(dim=1) + epsilon)

    # Weiszfeld iterations - solve weighted least-square problems.
    for iter in range(10):
        # Compute weights := inverse of reprojection error.
        dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
        w = (dis + epsilon).reciprocal()

        # Update with closed-form solution to weighted least-square problems.
        focal = (w * dot_xy_px).sum(dim=1) / ((w * dot_xy_xy).sum(dim=1) + epsilon)
    
    # Restrict field of view.
    focal_base = max(H, W) / (2 * (torch.deg2rad(torch.tensor(60, device=device)).tan() / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)

    # Return intrinsics.
    focal = rearrange(focal, '(b f) -> b f', b=B)
    intrinsics = torch.zeros(B, F, 3, 3, device=device)
    intrinsics[:,:,0,0] = intrinsics[:,:,1,1] = focal
    intrinsics[:,:,:2,2] = pp

    return intrinsics
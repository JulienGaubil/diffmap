import torch

from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor



def warp_image_flow(
        src_image: Float[Tensor, "3 height width"],
        flow: Float[Tensor, "vu=2 height width"]
) -> Float[Tensor, "3 height width"]:
    """Warp a source image into a target one using optical flow. Optical flow indexing is assumed to be 'vu': 
    p'[u + flow[u,v,1], v  + flow[u,v,0]] = p[u,v] where p pixel from source image, p' matching pixel from target image.
    """
    _, H, W = src_image.size()
    
    HW = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'), dim=0) #(2,H,W)

    # Reindex flow and undo normalization.
    flow = flow[[1,0],:,:] # (uv=2, height, width) indexing vu -> uv to match pixel grid indexing 
    hw = torch.tensor((H, W), dtype=torch.float32)
    flow = flow * hw[:,None,None]

    #### Manual way ####

    # Warp coordinates.
    warped_pixel_coordinates = torch.round(HW + flow).to(int) #(2,H,W)
    valid_mask = (warped_pixel_coordinates[0,:,:] >= 0) & \
            (warped_pixel_coordinates[0,:,:] < H) & \
            (warped_pixel_coordinates[1,:,:] >= 0) & \
            (warped_pixel_coordinates[1,:,:] < W) #(H,W)
    HW_warp = warped_pixel_coordinates[:,valid_mask] #(2,N)
    
    # Warp pixels.
    warped_image = torch.zeros(3, H, W)
    warped_image[:, HW_warp[0,:], HW_warp[1,:]] = src_image[:,valid_mask]


    ##### PyTorch way, cf https://github.com/dcharatan/flowmap/blob/main/flowmap/model/projection.py  ######

    # source_xy, _ = sample_image_grid((H, W))
    # target_xy = source_xy + rearrange(flow, "b f h w xy -> (b f) h w xy")
    # target_pixels = F.grid_sample(
    #             src_image,
    #             target_xy * 2 - 1,
    #             mode="bilinear",
    #             padding_mode="zeros",
    #             align_corners=False
    #     )

    return warped_image
import torch

from torch import Tensor
from jaxtyping import Float
from torchvision.transforms import functional as FT
import collections

from ldm.data.utils.camera import Intrinsics   


class CenterCropFlow:
    def __init__(self, crop_size: int | tuple[int] | list[int]) -> None:
        if isinstance(crop_size, collections.abc.Sequence):
            self.crop_size_H, self.crop_size_W = crop_size
        else:
            self.crop_size_H = self.crop_size_W = crop_size

    def __call__(self, flow: Float[Tensor, "xy=2 height width"]) -> Float[Tensor, "xy=2 height_cropped width_cropped"]: 
        # Undo flow normalization.
        _, H, W = flow.size()
        WH = torch.tensor((W, H), dtype=torch.float32)
        flow = flow * WH[:,None,None]

        # Crop flow.
        flow_cropped = FT.center_crop(flow, [self.crop_size_H, self.crop_size_W])
        
        # Normalize flow back.
        _, H_cropped, W_cropped = flow_cropped.size()
        WH_cropped = torch.tensor((W_cropped, H_cropped), dtype=torch.float32)
        flow_cropped = flow_cropped / WH_cropped[:,None,None]

        return flow_cropped


class ResizeIntrinsics:
    def __init__(self, new_size: int | tuple[int] | list[int]) -> None:
        self.new_size = new_size
    
    def __call__(self, intrinsics: Intrinsics) -> Intrinsics:
        # Compute new size
        H, W = intrinsics.resolution
        if isinstance(self.new_size, int): # fix resolution along minimal dimension, aspect ratio preserved
            aspect_ratio = H / W
            if H > W:
                H_new = self.new_size * aspect_ratio
                W_new = self.new_size
            else:
                H_new = self.new_size
                W_new = self.new_size / aspect_ratio
        else: # aspect ratio not preserved
            H_new, W_new = self.new_size

        # Rescale intrinsics.
        intrinsics.fx = intrinsics.fx * (W_new / W)
        intrinsics.fy = intrinsics.fy * (H_new / H)
        intrinsics.cx = intrinsics.cx * (W_new / W)
        intrinsics.cy = intrinsics.cy * (H_new / H)

        # Update resolution.
        intrinsics.resolution = [H_new, W_new]
        
        return intrinsics
    

class CenterCropIntrinsics:
    def __init__(self, crop_size: int | tuple[int] | list[int]) -> None:
        if isinstance(crop_size, collections.abc.Sequence):
            self.crop_size_H, self.crop_size_W = crop_size
        else:
            self.crop_size_H = self.crop_size_W = crop_size

    def __call__(self, intrinsics: Intrinsics) -> Intrinsics:
        # Compute crop along every dimension.
        H, W = intrinsics.resolution
        t_x = (W - self.crop_size_W) / 2
        t_y = (H - self.crop_size_H) / 2

        # Update principal point.
        intrinsics.cx = intrinsics.cx - t_x
        intrinsics.cy = intrinsics.cy - t_y

        # Update resolution.
        intrinsics.resolution = [self.crop_size_H, self.crop_size_W]
        
        return intrinsics
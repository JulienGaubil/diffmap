import torch

from typing import Dict
from jaxtyping import Float
from torch import Tensor
from omegaconf import ListConfig
from einops import rearrange
from torchvision import transforms
from torchvision.transforms import functional as FT

from ldm.util import instantiate_from_config
from ldm.data.utils.camera import Intrinsics


class ResizeFlow:
    def __init__(self, new_size: int | tuple[int,int]) -> None:

        if isinstance(new_size, tuple):
            self.new_size_H, self.new_size_W = new_size
        else:
            self.new_size_H = self.new_size_W = new_size

    def __call__(self, flow: Float[Tensor, "xy=2 height width"]) -> Float[Tensor, "xy=2 height_new width_new"]:
        flow_resized = FT.resize(
            flow,
            [self.new_size_H, self.new_size_W],
            interpolation=transforms.InterpolationMode.BILINEAR
        )

        return flow_resized
    

class CenterCropFlow:
    def __init__(self, crop_size: int | tuple[int,int]) -> None:

        if isinstance(crop_size, tuple):
            self.crop_size_H, self.crop_size_W = crop_size
        else:
            self.crop_size_H = self.crop_size_W = crop_size

    def __call__(
            self,
            flow: Float[Tensor, "xy=2 height width"]
    ) -> Float[Tensor, "xy=2 height_cropped width_cropped"]:
        # Undo flow normalization.
        _, h, w = flow.size()
        wh = torch.tensor((w, h), dtype=torch.float32)
        flow = flow * wh[:,None,None]

        # Crop flow.
        flow_cropped = FT.center_crop(flow, [self.crop_size_H, self.crop_size_W])
        
        # Normalize flow back.
        _, h_cropped, w_cropped = flow.size()
        wh_cropped = torch.tensor((w_cropped, h_cropped), dtype=torch.float32)
        flow_cropped = flow_cropped / wh_cropped[:,None,None]

        return flow_cropped
    

class ResizeDepth:
    def __init__(self, new_size: int | tuple[int,int]) -> None:

        if isinstance(new_size, tuple):
            self.new_size_H, self.new_size_W = new_size
        else:
            self.new_size_H = self.new_size_W = new_size

    def __call__(self, depth: Float[Tensor, "height width"]) -> Float[Tensor, "height_new width_new"]:
        depth_resized = FT.resize(
            depth[None],
            [self.new_size_H, self.new_size_W],
            interpolation=transforms.InterpolationMode.NEAREST
        )

        return depth_resized.squeeze()


class ResizeIntrinsics:
    def __init__(self, new_size: int | tuple[int,int]) -> None:

        if isinstance(new_size, tuple):
            self.new_size_H, self.new_size_W = new_size
        else:
            self.new_size_H = self.new_size_W = new_size

    def __call__(
        self,
        intrinsics: Intrinsics,
    ) -> Intrinsics:
        # Compute rescale ratios.
        H, W = intrinsics.resolution
        ratio_H = H / self.new_size_H
        ratio_W = W / self.new_size_W

        # Rescale intrinsics.
        intrinsics.fx = intrinsics.fx / ratio_W
        intrinsics.fy = intrinsics.fy / ratio_H
        intrinsics.cx = intrinsics.cx / ratio_W
        intrinsics.cy = intrinsics.cy / ratio_H

        # Update resolution.
        intrinsics.resolution = [self.new_size_H, self.new_size_W]
        
        return intrinsics
    

class CenterCropIntrinsics:
    def __init__(self, crop_size: int | tuple[int,int]) -> None:

        if isinstance(crop_size, tuple):
            self.crop_size_H, self.crop_size_W = crop_size
        else:
            self.crop_size_H = self.crop_size_W = crop_size

    def __call__(
        self,
        intrinsics: Intrinsics,
    ) -> Intrinsics:
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
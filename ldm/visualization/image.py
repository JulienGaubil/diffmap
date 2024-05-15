import numpy as np
import torch
import torchvision

from jaxtyping import Float, Int
from numpy import ndarray
from torch import Tensor
from PIL import Image
from einops import rearrange


def overlay_images(
        im1: Float[ndarray, "height width 3"] | Float[Tensor, "3 height width"],
        im2: Float[ndarray, "height width channel=3"] | Float[Tensor, "3 height width"],
        alpha_im1: float = 0.5
    ) -> Float[ndarray, "height width 3"] | Float[Tensor, "3 height width"]:
    """Blend two numpy images with transparency alpha_im1 for im1 and (1 - alpha_im1) for im2.
    Output pixel range is [0,1].
    """
    alpha_im2 = 1 - alpha_im1

    # Pixel range to [0,1].
    if im1.max() > 1:
        im1 = im1 / 255
    if im2.max() > 1:
        im2 = im2 / 255

    # Transform to RGBA PIL images.
    im1_rgba = torchvision.transforms.functional.to_pil_image(im1).convert("RGBA")
    im2_rgba = torchvision.transforms.functional.to_pil_image(im2).convert("RGBA")

    # Overlay with alpha blending.
    overlayed_image = Image.new("RGBA", im1_rgba.size)
    overlayed_image = Image.blend(overlayed_image, im1_rgba, alpha=1.0)
    overlayed_image = Image.blend(overlayed_image, im2_rgba, alpha=alpha_im2) # compose with alpha
    overlayed_image = np.asarray(overlayed_image.convert("RGB")).astype(int) # pixel range(0,255)

    # Output pixel range in [0,1].
    overlayed_image = overlayed_image / 255

    if isinstance(im1, Tensor):
        overlayed_image = torch.from_numpy(overlayed_image)
        overlayed_image = rearrange(overlayed_image, 'h w c -> c h w') # bring back to torch image format

    return overlayed_image
import torch
import numpy as np
import torchvision

from PIL import Image
from io import BytesIO
from torch import Tensor
from jaxtyping import Float
from torchvision.transforms import ToTensor, ToPILImage

to_tensor = ToTensor()
to_pil = ToPILImage()

def load_jpeg_im(byte_tensor: Float[Tensor, "byte"]) -> Float[Tensor, "3 height width"]:
    pil_im = Image.open(BytesIO(byte_tensor.numpy().tobytes()))
    return to_tensor(pil_im)

def dump_jpeg_im(image_tensor: Float[Tensor, "3 height width"]) -> Float[Tensor, "byte"]:
    jpeg_tensor = torchvision.io.encode_jpeg((image_tensor * 255).type(torch.uint8), quality=95)
    return jpeg_tensor

def frames_to_jpegs(frames: Float[Tensor, "batch 3 height width"]) -> list[Float[Tensor, "byte"]]:
    """
    Converts torch CxHxW frame tensors to 1D byte tensor of JPEG compressed image.
    """
    jpeg_tensors = list()
    for k in range(frames.size(0)):
        jpeg_tensors.append(dump_jpeg_im(frames[k]))

    return jpeg_tensors


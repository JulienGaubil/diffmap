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

class BatchRunningStd:
    def __init__(self):
        self.n = 0
        self.S1 = 0
        self.S2 = 0

    def update(self, batch):

        batch = batch.flatten().to(torch.float32)
        batch_size = len(batch)
        self.S1 += batch.sum()
        self.S2 += (batch ** 2).sum()
        self.n += batch_size

    def get_mean_std(self):
        running_mean = (1 / self.n) * self.S1
        running_std = ((1 / (self.n - 1)) * (self.S2 - (1 / self.n) * (self.S1 **2))).sqrt()
        
        return running_mean, running_std


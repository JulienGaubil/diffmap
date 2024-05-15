import Imath
import numpy
import torch
import OpenEXR as exr

from PIL import Image
from jaxtyping import Float
from torch import Tensor
from pathlib import Path


def load_exr(filepath: Path) -> Float[Tensor, "height width"] :
    # Load bytes.
    exrfile = exr.InputFile(filepath.as_posix())
    raw_bytes = exrfile.channel('B', Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = numpy.frombuffer(raw_bytes, dtype=numpy.float32)

    # Reshape in depth map.
    height = exrfile.header()['displayWindow'].max.y + 1 - exrfile.header()['displayWindow'].min.y
    width = exrfile.header()['displayWindow'].max.x + 1 - exrfile.header()['displayWindow'].min.x
    depth_map = numpy.copy(numpy.reshape(depth_vector, (height, width)))

    return torch.from_numpy(depth_map)
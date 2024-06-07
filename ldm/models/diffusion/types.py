from typing import Literal
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass


@dataclass
class Sample:
    x_noisy: Float[Tensor, "sample noisy_channel height width"] # xt
    t: int | list[int] # diffusion step
    x_recon: Float[Tensor, "sample noisy_channel height width"] | None = None # clean x0 estimate
    depths: Float[Tensor, "sample frame height width"] | None = None
    weights: Float[Tensor, "sample pair=frame-1 height width"] | None = None


@dataclass
class DiffusionOutput:
    diff_output: Float[Tensor, "sample noisy_channel height width"] # prediction for denoised modalities (x0 or eps depending on parameterization)
    clean: Float[Tensor, "sample clean_channel height width"] # non-denoised modalities
    weights: Float[Tensor, "sample pair height width"] | None = None
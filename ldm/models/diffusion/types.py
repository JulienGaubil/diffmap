from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass


@dataclass
class Sample:
    x_noisy: Float[Tensor, "batch channel height width"] # xt
    t: int | list[int] # diffusion step
    x_recon: Float[Tensor, "batch channel height width"] | None = None # clean x0 estimate
    depths: Float[Tensor, "batch frame=2 height width"] | None = None
    weights: Float[Tensor, "batch pair=1 height width"] | None = None


@dataclass
class DiffusionOutput:
    diff_output: Float[Tensor, "batch channel_noisy height width"] # prediction for denoised modalities (x0 or eps depending on parameterization)
    clean: Float[Tensor, "batch channel_clean height width"] # non-denoised modalities
    weights: Float[Tensor, "batch pair=1 height width"] | None = None
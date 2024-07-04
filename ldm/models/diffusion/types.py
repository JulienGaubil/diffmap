from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass

@dataclass
class Sample:
    t: int | list[int] # diffusion step
    x_denoised: Float[Tensor, "sample noisy_channel height width"] # xt
    x_predicted: Float[Tensor, "sample clean_channel height width"] | None = None # non-denoised modalities
    x_recon: Float[Tensor, "sample noisy_channel height width"] | None = None # clean x0 estimate
    weights: Float[Tensor, "sample pair height width"] | None = None
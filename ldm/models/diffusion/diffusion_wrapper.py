import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from omegaconf import OmegaConf
from dataclasses import dataclass

import torch.nn.functional as F
from ldm.misc.util import instantiate_from_config
from ldm.thirdp.flowmap.flowmap.model.backbone.backbone_midas import make_net
from ldm.misc.modalities import Modalities


#Class wrapper du U-Net, appelle son forward pass dans forward
class DiffusionMapWrapper(pl.LightningModule):
    def __init__(
        self,
        diff_model_config: OmegaConf,
        conditioning_key: str,
        image_size: int,
        modalities_out: Modalities,
        compute_weights: bool = False,
        latent: bool = False,
        n_future: int = 1,
        **kwargs
    ) -> None:
        super().__init__()

        self.conditioning_key = conditioning_key
        self.compute_weights = compute_weights
        self.latent = latent
        self.n_future = n_future
        self.modalities_out = modalities_out
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm']

        # Instantiate diffusion model.
        self.diffusion_model = instantiate_from_config(diff_model_config)

        # Instantiate output heads.
        self.diff_out = self.diffusion_model.out
        self.diffusion_model.out = nn.Identity()
        if self.compute_weights:
            model_channels = self.diffusion_model.model_channels
            if self.latent:
                image_size = 8 * image_size # TODO properly handle upsampling

            self.corr_weighter_perpoint = make_net([model_channels, image_size, model_channels, self.n_future])

    def forward(
        self,
        x: Float[Tensor, "sample (frame channel) height width"],
        t: Int[Tensor, "sample"],
        c_concat: list[Float[Tensor, "batch channel height width"]] | None = None,
        c_crossattn: list[Float[Tensor, '...']] | None = None,
        c_adm: None = None
    ) -> tuple[Float[Tensor, "batch channel height width"] | None]:
        
        #U-Net forward pass for N-1 layers.
        if self.conditioning_key is None:
            features = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            features = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            features = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            features = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            features = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            features = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        # Compute diffusion output.
        diff_out = self.diff_out(features)

        # Separate noisy and clean modalities.
        denoised, clean = self.modalities_out.split_noisy_clean(diff_out)

        # Compute correspondence weights. 
        if self.compute_weights:
            weights = self.compute_correspondence_weights(features)
        else:
            weights = None

        return denoised, clean, weights
    
    def compute_correspondence_weights(
        self,
        features: Float[Tensor, "sample channel height width"],
    ) -> Float[Tensor, "sample pair height width"]:
        b, _, h, w = features.shape
        if self.latent: # TODO properly handle weight computation in latent space
            features = F.interpolate(features, (h*8, w*8), mode="bilinear")
        features = rearrange(features, "b c h w -> b () h w c")

        # Compute correspondence weights.
        weights = self.corr_weighter_perpoint(features).sigmoid().clip(min=1e-4)
        weights = rearrange(weights, "b () h w p -> b p h w")
        
        return weights
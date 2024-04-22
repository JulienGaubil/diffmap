import torch
import torch.nn as nn
import pytorch_lightning as pl
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor
from dataclasses import dataclass

import torch.nn.functional as F
from ldm.util import instantiate_from_config
from ldm.modules.flowmap.model.backbone.backbone_midas import make_net


@dataclass
class DiffusionOutput:
    diff_output: Float[Tensor, "batch channel_noisy height width"] # prediction for denoised modalities (x0 or eps depending on parameterization)
    clean: Float[Tensor, "batch channel_clean height width"] # non-denoised modalities
    weights: Float[Tensor, "batch pair=1 height width"] | None = None


#Class wrapper du U-Net, appelle son forward pass dans forward
class DiffusionMapWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key, image_size, compute_weights=False, latent=False):
        super().__init__()

        self.diffusion_model = instantiate_from_config(diff_model_config) # U-Net
        self.diff_out = self.diffusion_model.out
        self.diffusion_model.out = nn.Identity()

        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm']

        self.compute_weights = compute_weights
        self.latent = latent
        if self.compute_weights:
            model_channels = self.diffusion_model.model_channels
            if self.latent:
                image_size = 8 * image_size # TODO properly handle upsampling

            self.corr_weighter_perpoint = make_net([model_channels, image_size, model_channels, 1])

    def forward(self,
                x: Float[Tensor, "batch channel height width"],
                t: Int[Tensor, "batch"],
                c_concat: list[Float[Tensor, "batch channel height width"]] | None = None,
                c_crossattn: list[Float[Tensor, '...']] | None = None,
                c_adm: None = None
            ) -> DiffusionOutput:
        #U-Net forward pass with potential correspondence weights estimation.
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

        # Compute diffusion module output
        c_noisy = x.size(1)
        diff_out = self.diff_out(features)
        c_clean =  diff_out.size(1) - c_noisy
        denoised, clean = diff_out.split([c_noisy, c_clean], dim=1)
        out = DiffusionOutput(denoised, clean)

        # Compute correspondence weights. 
        if self.compute_weights:
            out.weights = self.compute_correspondence_weights(features)
        return out
    
    def compute_correspondence_weights(
        self,
        features: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, "batch frame=1 height width"]:
        b, _, h, w = features.shape
        if self.latent: # TODO properly handle weight computation in latent space
            features = F.interpolate(features, (h*8, w*8), mode="bilinear")
        features = rearrange(features, "(b f) c h w -> b f h w c", b=b, f=1) # considers pairs of frames
        weights = self.corr_weighter_perpoint(features).sigmoid().clip(min=1e-4)
        return rearrange(weights, "b f h w () -> b f h w")
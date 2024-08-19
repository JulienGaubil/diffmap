import torch
import torch.nn as nn
import pytorch_lightning as pl

from einops import rearrange
from jaxtyping import Float, Int
from typing import Literal
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass

import torch.nn.functional as F
from ldm.misc.util import instantiate_from_config
from ldm.thirdp.flowmap.flowmap.model.backbone.backbone_midas import make_net
from ldm.misc.modalities import Modalities, GeometryModalities
from .model_wrapper import ModelWrapper

# @dataclass
# class UNetCfg:
#     image_size: int
#     model_channels: int
#     attention_resolutions: list[int]
#     num_res_blocks: int
#     channel_mult: list[int]
#     num_heads: int
#     use_spatial_transformer: bool
#     transformer_depth: int
#     context_dim: int | None
#     use_checkpoint: bool
#     legacy: bool


# @dataclass
# class UNetWrapperCfg:
#     model_cfg: OmegaConf
#     conditioning_key: str
#     image_size: int
#     compute_weights: bool = False
#     latent: bool = False
#     n_future: int = 1


#Class wrapper du U-Net, appelle son forward pass dans forward
class UNetWrapper(ModelWrapper):
    def __init__(
        self,
        modalities_in: Modalities,
        modalities_out: Modalities,
        model_cfg: DictConfig,
        image_size: int,
        conditioning_key: Literal["crossattn", "concat", "hybrid"] | None = None, #defines the conditioning method
        compute_weights: bool = False,
        latent: bool = False,
        **kwargs
    ) -> None:

        assert modalities_out.n_noisy_channels == modalities_in.n_noisy_channels
        model_cfg.params.in_channels = modalities_in.n_channels #enforce number of input channels for model
        model_cfg.params.out_channels = modalities_out.n_channels #enforce number of input channels for model
        super().__init__(modalities_in, modalities_out, model_cfg)

        self.conditioning_key = conditioning_key
        self.compute_weights = compute_weights
        self.latent = latent

        # Instantiate output heads.
        self.diff_out = self.diffusion_model.out
        self.diffusion_model.out = nn.Identity()

        # MLP for weights regression from diffusion features.
        if self.compute_weights:
            model_channels = self.diffusion_model.model_channels
            
            geometric_modalities = [subset for subset in self.modalities_out.subsets if isinstance(subset, GeometryModalities)]
            #TODO - remove hack
            assert len(geometric_modalities) == 1
            n_future = geometric_modalities[0]._future_modality.multiplicity if geometric_modalities[0]._future_modality is not None else 0
            assert n_future > 0

            if self.latent:
                image_size = 8 * image_size # TODO properly handle upsampling
            self.corr_weighter_perpoint = make_net([model_channels, image_size, model_channels, n_future])

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
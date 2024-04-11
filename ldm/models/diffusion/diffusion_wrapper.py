import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager, nullcontext
from functools import partial
import itertools
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
from omegaconf import ListConfig

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.attention import CrossAttention

from jaxtyping import Float, Int
from torch import Tensor
from ldm.modules.flowmap.model.model_wrapper_pretrain import FlowmapLossWrapper
from ldm.modules.flowmap.config.common import get_typed_root_config_diffmap
from ldm.modules.flowmap.config.pretrain import DiffmapCfg
from ldm.modules.flowmap.loss import get_losses
from ldm.modules.flowmap.model.model import FlowmapModelDiff
from ldm.modules.flowmap.model.projection import earlier, later, sample_image_grid
from ldm.modules.flowmap.model.backbone.backbone_midas import make_net
from ldm.modules.flowmap.flow.flow_predictor import Flows

from dataclasses import dataclass


@dataclass
class DiffusionOutput:
    denoised: Float[Tensor, "batch channel height width"]
    weights: Float[Tensor, "batch height width"] | None = None


#Class wrapper du U-Net, appelle son forward pass dans forward
class DiffusionMapWrapper(pl.LightningModule, ):
    def __init__(self, diff_model_config, conditioning_key, image_size, compute_weights=False):
        super().__init__()

        self.diffusion_model = instantiate_from_config(diff_model_config) # U-Net
        self.diff_out = self.diffusion_model.out
        self.diffusion_model.out = nn.Identity()

        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm']

        self.compute_weights = compute_weights
        if self.compute_weights:
            model_channels = self.diffusion_model.model_channels
            # TODO change 128 to (cropping//patch_size)*patch_size
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
        b, _, h, w = features.size()
        # out = rearrange(self.diff_out(features), "(b f) c h w -> b f c h w", b=b, f=1) # considers pairs of frames
        out = self.diff_out(features)
        
        # Compute correspondence weights. 
        if self.compute_weights:
            backward_weights = self.compute_correspondence_weights(features)
            return DiffusionOutput(out, backward_weights.squeeze(1))
        else:
            return DiffusionOutput(out)

    def compute_correspondence_weights(
        self,
        features: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, "batch frame=1 height width"]:
        b, _, h, w = features.size()
        features = rearrange(features, "(b f) c h w -> b f h w c", b=b, f=1) # considers pairs of frames
        # print('SIZE FEATURES WEIGHTS : ', features.size())
        # print('WEIGHTS MODEL : ')
        # print(self.corr_weighter_perpoint)
        weights = self.corr_weighter_perpoint(features).sigmoid().clip(min=1e-4)
        return rearrange(weights, "b f h w () -> b f h w")
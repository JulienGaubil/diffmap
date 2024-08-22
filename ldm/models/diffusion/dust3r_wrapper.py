import sys
sys.path.append('ldm/thirdp/dust3r')

import os
import copy
import torch

from typing import Dict, Literal
from jaxtyping import Int, Float
from torch import Tensor
from einops import repeat, rearrange
from omegaconf import OmegaConf, DictConfig

from .model_wrapper import ModelWrapper
from ldm.modules.diffusionmodules.dust3rmodel import Dust3rModel
from ldm.misc.modalities import RGBModalities
from ldm.misc.modalities import Modalities, RGBModalities, GeometryModalities, SequenceModalities
from ldm.thirdp.dust3r.dust3r.utils.misc import transpose_to_landscape
from ldm.thirdp.dust3r.dust3r.heads import head_factory
from ldm.thirdp.dust3r.dust3r.model import AsymmetricCroCo3DStereo, load_model

inf = float('inf')


class Dust3rWrapper(ModelWrapper):
    def __init__(
        self,
        modalities_in: Modalities,
        modalities_out: Modalities,
        model_cfg: DictConfig,
        conditioning_key: str = 'crossattn',
        ckpt_path: str | None = None,
        **kwargs
    ) -> None:

        self.conditioning_key = conditioning_key
        
        # Instantiate wrapper and model.
        assert modalities_out.n_noisy_channels == modalities_in.n_noisy_channels
        # TODO - remove hack 
        assert any(
            isinstance(subset, RGBModalities) and subset._future_modality is not None and subset._future_modality.multiplicity == 1
            for subset in modalities_in.subsets
        )
        assert any(
            isinstance(subset, GeometryModalities) and subset._future_modality is not None and subset._future_modality.multiplicity == 1 \
            and subset._past_modality is not None and subset._past_modality.multiplicity == 1
            for subset in modalities_out.subsets
        )
        geometric_modalities = [modalities_group for modalities_group in modalities_out.subsets if isinstance(modalities_group, GeometryModalities)]
        assert len(geometric_modalities) == 1
        
        super().__init__(modalities_in, modalities_out, model_cfg)
        
        # Load checkpoint.
        if ckpt_path is not None:
            if os.path.isfile(ckpt_path):
                self.load_checkpoint(ckpt_path, device='cpu')
        
        # Instantiate heads.
        for subset in self.modalities_out.subsets:
            if isinstance(subset, SequenceModalities):

                for modality, original_head in zip([subset._past_modality, subset._future_modality], [self.diffusion_model.downstream_head1, self.diffusion_model.downstream_head2]):
                    if modality is not None:
                        head_id = '_'.join(['head', modality._id])

                        if isinstance(subset, GeometryModalities):
                            head = copy.deepcopy(original_head)
                        elif isinstance(subset, RGBModalities):
                            head = head_factory('dpt', 'pts3d', self.diffusion_model, has_conf=False)
                            head.conf_mode = None
                            head.depth_mode = ('linear', -float('inf'), float('inf'))
                        else:
                            raise NotImplementedError()

                        # Instantiate heads.
                        setattr(
                            self.diffusion_model,
                            'downstream_' + head_id,
                            head
                        )
                        setattr(
                            self.diffusion_model,
                            head_id,
                            transpose_to_landscape(getattr(self.diffusion_model, 'downstream_' + head_id), activate=True)
                        )
            else:
                raise NotImplementedError()
        # Replace original heads.
        self.diffusion_model.head1 = lambda *x: x
        self.diffusion_model.head2 = lambda *x: x
        del self.diffusion_model.downstream_head1
        del self.diffusion_model.downstream_head2

    def load_checkpoint(self, ckpt_path: str, device: str = 'cpu', verbose: bool = True) -> None:
        if verbose:
            print('... loading model from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        if 'landscape_only' not in args:
            args = args[:-1] + ', landscape_only=False)'
        else:
            args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
        assert "landscape_only=False" in args
        if verbose:
            print(f"instantiating : {args}")
        s = self.diffusion_model.load_state_dict(ckpt['model'], strict=False)
        if verbose:
            print(s)
        self.diffusion_model.to(torch.device(device))

    def prepare_input(
        self,
        x_noisy: Float[Tensor, "sample noisy_channel height width"],
        x_clean: list[Float[Tensor, "sample clean_channel height width"]],
    ) -> tuple[Dict,Dict]:
        """Dust3r Input formatting.
        """
        x_clean = torch.cat(x_clean, 1)
        B, _, H, W = x_clean.shape

        # TODO - remove hack
        assert x_noisy.size(1) == 3 and x_clean.size(1) == 3

        true_shape1 = torch.tensor([H, W], dtype=torch.int32)
        true_shape2 = torch.tensor([H, W], dtype=torch.int32)
        true_shape1 = repeat(true_shape1, 'hw -> b hw', b=x_clean.size(0))
        true_shape2 = repeat(true_shape2, 'hw -> b hw', b=x_clean.size(0))

        view1 = dict(
            img=x_clean,
            true_shape=true_shape1,
            idx=[k for k in range(B)],
            instance=[str(k) for k in range(B)]
        )
        view2 = dict(
            img=x_noisy,
            true_shape=true_shape2,
            idx=[k for k in range(B)],
            instance=[str(k) for k in range(B)]
        )

        return view1, view2

    def prepare_output(
        self,
        dust3r_outputs: Dict[str, Float[Tensor, "sample height width channel"]]
    ) -> tuple[
        Float[Tensor, "sample noisy_channel height width"],
        Float[Tensor, "sample clean_channel height width"],
        Float[Tensor, "sample frame=2 height width"] | None,
    ]:
        """Diffmap output formatting.
        """
        
        ccat_dict = dict()
        weights = list()
        for subset in self.modalities_out.subsets:
            for modality in [subset._past_modality, subset._future_modality]:
                if modality is not None:
                    ccat_dict[modality._id] = dust3r_outputs[modality._id]['pts3d']
                    if isinstance(subset, GeometryModalities):
                        weights.append(dust3r_outputs[modality._id]['conf'])

        # Separate noisy and clean modalities.
        diff_out = self.modalities_out.cat_modalities(ccat_dict, dim=-1)
        diff_out = rearrange(diff_out, 'b h w c -> b c h w')
        denoised, clean = self.modalities_out.split_noisy_clean(diff_out)
        weights = torch.stack(weights, dim=1) if len(weights) > 0 else None

        return denoised, clean, weights

    def forward(
        self,
        x: Float[Tensor, "sample (frame noisy_channel) height width"],
        t: Int[Tensor, "sample"],
        c_crossattn: list[Float[Tensor, "sample (frame clean_channel) height width"]],
        **kwargs
    ) -> tuple[
        Float[Tensor, "sample noisy_channel height width"],
        Float[Tensor, "sample clean_channel height width"],
        Float[Tensor, "sample height width"] | None,
    ]:
        view1, view2 = self.prepare_input(x, c_crossattn)

        t_clean = torch.zeros_like(t)

        dec1, dec2 = self.diffusion_model(view1, view2, t_clean, t)

        # Apply output heads.
        dust3r_outputs = dict()
        with torch.cuda.amp.autocast(enabled=False):
            for subset in self.modalities_out.subsets:
                for modality, dec_output in zip(
                    [subset._past_modality, subset._future_modality],
                    [dec1, dec2],
                ):
                    if modality is not None:
                        id_m = modality._id
                        head = getattr(self.diffusion_model, f'head_{id_m}')
                        dust3r_outputs[id_m] = head(*dec_output)

        denoised, clean, weights = self.prepare_output(dust3r_outputs)
        return denoised, clean, weights
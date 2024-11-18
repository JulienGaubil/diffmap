import torch
import torch.nn as nn

from torch import Tensor
from jaxtyping import Float, Int
from typing import Any, Dict, Tuple
from einops import rearrange, repeat
from omegaconf import OmegaConf, DictConfig

from ldm.misc.modalities import Modalities
from ldm.models.diffusion.ddpm_diffmap import DDPMDiffmap


class DiffusionLoss(nn.Module):
    learn_logvar: bool
    def __init__(
        self,
        loss_cfg: DictConfig,
    ) -> None:
        super().__init__()

        self.l_simple_weight = loss_cfg.l_simple_weight
        self.original_elbo_weight = loss_cfg.original_elbo_weight

        self.loss_type = loss_cfg.loss_type

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def prepare_loss_inputs(
        self,
        output_dict: Dict[str, Float[Tensor, "batch frame channel height width"]],
        batch: Dict,
        modalities_in: Modalities,
        modalities_out: Modalities,
        t: Int[Tensor, f"sample"],
        noise: Float[Tensor, "batch frame channel height width"],
        diffusion_module: DDPMDiffmap,
    ) -> Tuple[Dict, Dict, Modalities, Int[Tensor, f"sample"], nn.Module]:
        assert modalities_in.n_noisy_channels > 0 and modalities_out.n_noisy_channels > 0, "Denoising loss used without noisy I/O."
        assert modalities_in.denoised_modalities == modalities_out.denoised_modalities

        # Prepare diffusion loss inputs.
        if diffusion_module.parameterization == "x0":
            denoising_target = batch['x_start']
        elif diffusion_module.parameterization == "eps":
            denoising_target = noise
        else:
            raise NotImplementedError()
        
        denoised_output_dict = {id:rearrange(output_dict[id], 'b f c h w -> (b f) c h w') for id in modalities_out.ids_denoised}
        target_output_dict = modalities_out.split_modalities(denoising_target, modality_ids=modalities_out.ids_denoised)
        target_output_dict = {k:rearrange(v, 'b (f c) h w -> (b f) c h w', c=diffusion_module.channels_m) for k,v in target_output_dict.items()}

        return denoised_output_dict, target_output_dict, modalities_out, t, diffusion_module

    def criterion(
        self,
        denoised_output_dict: Dict,
        target_output_dict: Dict,
        modalities_out: Modalities,
        t: Int[Tensor, f"sample"],
        diffusion_module: DDPMDiffmap,
    ) -> Tuple[float, Dict]:
        device = diffusion_module.device

        loss_simple, loss_gamma, loss, loss_vlb  = 0, 0, 0, 0
        metrics_dict = dict()
        prefix = "train" if diffusion_module.training else "val"

        logvar_t = diffusion_module.logvar.to(device)
        logvar_t = repeat(logvar_t[t], 'b -> b f', f=diffusion_module.n_future)
        logvar_t = rearrange(logvar_t, 'b f -> (b f)')
        if diffusion_module.learn_logvar:
            metrics_dict.update({'logvar': diffusion_module.logvar.data.mean()})

        for modality in modalities_out.denoised_modalities:
            id_m = modality._id

            # Simple diffusion loss.
            loss_simple_m = self.get_loss(denoised_output_dict[id_m], target_output_dict[id_m], mean=False).mean([1, 2, 3]) # TODO - change get_loss
            loss_simple += loss_simple_m
            metrics_dict.update({f'{prefix}_{id_m}/loss_simple': loss_simple_m.clone().detach().mean()})
            loss_gamma_m = loss_simple_m / torch.exp(logvar_t) + logvar_t
            if diffusion_module.learn_logvar:
                metrics_dict.update({f'{prefix}_{id_m}/loss_gamma': loss_gamma_m.mean()})
            loss_gamma += loss_gamma_m

            # VLB loss.
            loss_vlb_m = self.get_loss(denoised_output_dict[id_m], target_output_dict[id_m], mean=False).mean(dim=(1, 2, 3)) # TODO - change get_loss
            lvlb_weights = repeat(diffusion_module.lvlb_weights[t], 'b -> b f', f=modality.multiplicity)
            lvlb_weights = rearrange(lvlb_weights, 'b f -> (b f)')
            loss_vlb_m = (lvlb_weights * loss_vlb_m).mean()
            metrics_dict.update({f'{prefix}_{id_m}/loss_vlb': loss_vlb_m})
            loss_vlb += loss_vlb_m

            # Total loss.
            loss_m = self.l_simple_weight * loss_gamma_m.mean() + (self.original_elbo_weight * loss_vlb_m)
            metrics_dict.update({f'{prefix}_{id_m}/loss': loss_m})
            loss += loss_m
    
        # Log total diffusion losses.
        metrics_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        if diffusion_module.learn_logvar:
            metrics_dict.update({f'{prefix}/loss_gamma': loss_gamma.mean()})
        metrics_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        return loss, metrics_dict

    def forward(
        self,
        output_dict: Dict[str, Float[Tensor, "batch frame channel height width"]],
        modalities_in: Modalities,
        modalities_out: Modalities,
        batch: Dict,
        prefix: str,
        t: Int[Tensor, f"sample"],
        noise: Float[Tensor, "batch frame channel height width"],
        diffusion_module: DDPMDiffmap,
        **kwargs
    ) -> Tuple[float, Dict]:
        denoised_output_dict, target_output_dict, modalities_out, t, diffusion_module = self.prepare_loss_inputs(output_dict, batch, modalities_in, modalities_out, t, noise, diffusion_module)
        
        loss, metrics_dict = self.criterion(
            denoised_output_dict,
            target_output_dict,
            modalities_out,
            t,
            diffusion_module
        )
        metrics_dict.update({f'{prefix}/diffusion_loss': loss})
        
        return loss, metrics_dict
import torch
import torch.nn as nn
import os, sys
sys.path.append(os.path.abspath("ldm/thirdp/dust3r"))

from torch import Tensor
from jaxtyping import Float
from typing import Any, Dict, Tuple
from einops import rearrange
from omegaconf import OmegaConf, DictConfig

from ldm.thirdp.dust3r.dust3r.losses import L21Loss, Regr3D, ConfLoss
from ldm.misc.modalities import Modalities, GeometryModalities


class Dust3rLoss(nn.Module):
    def __init__(
        self,
        loss_cfg: DictConfig,
    ) -> None:
        super().__init__()
        pixel_loss = L21Loss()
        loss_3D_params = OmegaConf.to_container(loss_cfg.loss_3D)
        loss_3D = Regr3D(pixel_loss, **loss_3D_params)
        self.criterion = ConfLoss(loss_3D, loss_cfg.alpha)

    def prepare_loss_inputs(
        self,
        output_dict: Dict[str, Float[Tensor, "batch frame channel height width"]],
        batch: Dict,
        modalities_in: Modalities,
        modalities_out: Modalities
    ) -> Dict:
        geometric_modalities = [subset for subset in modalities_out.subsets if isinstance(subset, GeometryModalities)]
        assert len(geometric_modalities) == 1
        geometric_modalities = geometric_modalities[0]
        assert geometric_modalities.parameterization == "shared_pointmap"
        assert geometric_modalities._past_modality is not None and geometric_modalities._future_modality is not None

        # TODO - remove hack
        assert geometric_modalities._past_modality.multiplicity == 1 and geometric_modalities._future_modality.multiplicity == 1
        pts3d1 = geometric_modalities.to_geometry(output_dict[geometric_modalities._past_modality._id])
        pts3d2 = geometric_modalities.to_geometry(output_dict[geometric_modalities._future_modality._id])
        pts3d1 = rearrange(pts3d1, 'b () c h w -> b h w c')
        pts3d2 = rearrange(pts3d2, 'b () c h w -> b h w c')

        conf = output_dict['conf'] # (B, 2, H, W)
        valid_masks = output_dict['valid_mask'] # (B, 2, H, W)
        conf1, conf2 = conf.chunk(2, dim=1)
        valid_mask1, valid_mask2 = valid_masks.chunk(2, dim=1)
        conf1 = conf1.squeeze(1)
        conf2 = conf2.squeeze(1)
        valid_mask1 = valid_mask1.squeeze(1) # (B,H,W) - bool
        valid_mask2 = valid_mask2.squeeze(1) # (B,H,W) - bool

        B, H, W, _ = pts3d1.shape
        device = pts3d1.device

        gt1=dict(
            camera_pose=batch['ctxt_camera_pose'], # (B,4,4)
            pts3d=batch[geometric_modalities._past_modality._id], # (B,H,W,3)
            valid_mask=valid_mask1, # (B,H,W) - bool
        )
        
        gt2=dict(
            camera_pose=batch['trgt_camera_pose'], # (B,4,4)
            pts3d=batch[geometric_modalities._future_modality._id], # (B,H,W,3)
            valid_mask=valid_mask2,
        )
        pred1=dict(
            pts3d=pts3d1,  # (B,H,W,3)
            conf=conf1 # (B,H,W)
        )

        pred2=dict(
            pts3d_in_other_view=pts3d2,  # (B,H,W,3)?
            conf=conf2 # (B,H,W)
        )
      
        return gt1, gt2, pred1, pred2

    def forward(
        self,
        output_dict: Dict[str, Float[Tensor, "batch frame channel height width"]],
        modalities_in: Modalities,
        modalities_out: Modalities,
        batch: Dict,
        prefix: str,
        **kwargs
    ) -> Tuple[float, Dict]:
        gt1, gt2, pred1, pred2 = self.prepare_loss_inputs(output_dict, batch, modalities_in, modalities_out)
        
        loss, metrics_dict = self.criterion(gt1, gt2, pred1, pred2)

        metrics_dict.update({f'{prefix}/dust3r_loss': loss})
        return loss, metrics_dict
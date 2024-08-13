import os, sys
sys.path.append(os.path.abspath("ldm/thirdp/dust3r"))

from torch import Tensor
from jaxtyping import Float
from typing import Any, Dict
from einops import rearrange
from omegaconf import OmegaConf, DictConfig

from ldm.thirdp.dust3r.dust3r.losses import *
from ldm.misc.modalities import Modalities, GeometryModalities


class Dust3rLoss(nn.Module):
    def __init__(
        self,
        loss_cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.cfg = loss_cfg
        self.init()
    
    def init(
        self
    ) -> None:
        pixel_loss = L21Loss()
        loss_3D_params = OmegaConf.to_container(self.cfg.loss_3D)
        loss_3D = Regr3D(pixel_loss, **loss_3D_params)

        self.criterion = ConfLoss(loss_3D, self.cfg.alpha)

    def prepare_loss_inputs(
        self,
        predicted_dict: Dict[str,Float[Tensor, "sample clean_channels height width"]],
        conf: Float[Tensor, "sample frame height width"],
        gt1: Dict,
        gt2: Dict,
        geometric_modalities: GeometryModalities,
        **kwargs
    ) -> Any:
        # Should be already done
        #predicted = rearrange(predicted, 'b (f c) h w -> b f c h w', c=self.channels_m)
        #predicted_split_all = self.modalities_out.split_modalities_multiplicity(predicted, modality_ids=self.modalities_out.ids_clean) # TODO - remove hack, assumes clean modalities = depths only

        assert geometric_modalities.parameterization == "shared_pointmap"
        assert geometric_modalities._past_modality is not None and geometric_modalities._future_modality is not None

        # TODO - remove hack
        assert geometric_modalities._past_modality.multiplicity == 1 and geometric_modalities._future_modality.multiplicity == 1
        pts3d1 = geometric_modalities.to_geometry(predicted_dict[geometric_modalities._past_modality._id])
        pts3d2 = geometric_modalities.to_geometry(predicted_dict[geometric_modalities._future_modality._id])
        pts3d1 = rearrange(pts3d1, 'b f c h w -> b (f c) h w')
        pts3d2 = rearrange(pts3d2, 'b f c h w -> b (f c) h w')
        conf1, conf2 = conf.chunk(2, dim=1)
        conf1 = conf1.squeeze(1)
        conf2 = conf2.squeeze(1)

        loss_inputs = dict(
            gt1=gt1,
            gt2=gt2,
            pred1=dict(
                pts3d=pts3d1,  # (B,H,W,3)
                conf=conf1 # (B,H,W)
            ),
            pred2=dict(
                pts3d_in_other_view=pts3d2,  # (B,H,W,3)?
                conf=conf2 # (B,H,W)
            )
                
        )
      
        return loss_inputs

    def forward(
        self,
        predicted_dict,
        conf,
        **kwargs
    ) -> Any:
        inputs = self.prepare_loss_inputs(predicted_dict, conf, **kwargs)
        return self.criterion(**inputs)
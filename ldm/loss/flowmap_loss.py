import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float
from typing import Tuple, Type, TypeVar, Dict
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig

from ldm.misc.projection import intrinsics_from_3D
from ldm.misc.modalities import Modalities, GeometryModalities
from ldm.thirdp.flowmap.flowmap.model.intrinsics import IntrinsicsCfg, get_intrinsics
from ldm.thirdp.flowmap.flowmap.model.extrinsics import ExtrinsicsCfg, get_extrinsics
from ldm.thirdp.flowmap.flowmap.model.model import ModelOutput
from ldm.thirdp.flowmap.flowmap.flow.flow_predictor import Flows
from ldm.thirdp.flowmap.flowmap.dataset.types import Batch
from ldm.thirdp.flowmap.flowmap.model.backbone.backbone import BackboneOutput
from ldm.thirdp.flowmap.flowmap.model.projection import sample_image_grid, unproject
from ldm.thirdp.flowmap.flowmap.misc.cropping import CroppingCfg, crop_and_resize_batch_for_model
from ldm.thirdp.flowmap.flowmap.loss import Loss, LossCfg, get_losses
from ldm.thirdp.flowmap.flowmap.config.tools import get_typed_config, separate_multiple_defaults


T = TypeVar("T")

def get_typed_root_config_diffmap(cfg_dict: DictConfig, cfg_type: Type[T]) -> T:
    return get_typed_config(
        cfg_type,
        cfg_dict,
        {
            list[LossCfg]: separate_multiple_defaults(LossCfg),
        },
    )


@dataclass
class FlowmapWrapperCfg:
    intrinsics: IntrinsicsCfg
    extrinsics: ExtrinsicsCfg
    use_correspondence_weights: bool


@dataclass
class CommonDiffmapCfg:
    model: FlowmapWrapperCfg
    loss: list[LossCfg]
    cropping: CroppingCfg


@dataclass
class FlowmapLossCfg:
    patch_size: int


# TODO - remove?
@dataclass
class DiffmapCfg(CommonDiffmapCfg):
    model_wrapper: FlowmapLossCfg


class FlowmapWrapper(nn.Module):
    def __init__(
        self,
        cfg: FlowmapWrapperCfg,
        num_frames: int | None = None,
        image_shape: Tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.intrinsics = get_intrinsics(cfg.intrinsics)
        self.extrinsics = get_extrinsics(cfg.extrinsics, num_frames)

    def forward(
        self,
        batch: Batch,
        flows: Flows,
        depths: Float[Tensor, "batch frame height width"],
        correspondence_weights: Float[Tensor, "batch pair height width"],
        global_step: int,
        surfaces: Float[Tensor, "batch frame height width 3"] | None,
        intrinsics: Float[Tensor, "batch frame 3 3"] | None,
    ) -> ModelOutput:
        assert not (surfaces is not None and intrinsics is None), "No intrinsics provided along with pointmaps in Flowmap loss."

        device = batch.videos.device
        _, _, _, h, w = batch.videos.shape
        backbone_out = BackboneOutput(depths=depths, weights=correspondence_weights) # replaces depth estimator

        # Allow the correspondence weights to be ignored as an ablation.
        if not self.cfg.use_correspondence_weights:
            backbone_out.weights = torch.ones_like(backbone_out.weights)
        
        # Compute the intrinsics.
        if intrinsics is None:
            intrinsics = self.intrinsics.forward(batch, flows, backbone_out, global_step)
        
        if surfaces is None:
            # Use the intrinsics to calculate camera-space surfaces (point clouds).
            xy, _ = sample_image_grid((h, w), device=device)
            surfaces = unproject(
                xy,
                backbone_out.depths,
                rearrange(intrinsics, "b f i j -> b f () () i j"),
            )

        # Finally, compute the extrinsics.
        extrinsics = self.extrinsics.forward(batch, flows, backbone_out, surfaces)
        
        return ModelOutput(
            backbone_out.depths,
            surfaces,
            intrinsics,
            extrinsics,
            backbone_out.weights,
        )


class FlowmapLoss(nn.Module):
    def __init__(
        self,
        loss_cfg: DictConfig,
    ) -> None:
        super().__init__()

        cfg = get_typed_root_config_diffmap(loss_cfg, DiffmapCfg)
        self.cfg = cfg.model_wrapper
        self.cfg_cropping = cfg.cropping
        self.model = FlowmapWrapper(cfg.model)
        self.losses = get_losses(cfg.loss)

    def prepare_loss_inputs(
        self,
        output_dict: Dict[str, Float[Tensor, "batch frame channel height width"]],
        batch: Dict,
        modalities_in: Modalities,
        modalities_out: Modalities,
    ) -> tuple[
        dict[str, Tensor],
        Flows,
        Float[Tensor, "sample frame height width"],
        Float[Tensor, "sample pair height width"],
        Float[Tensor, "batch frame height width 3"] | None,
        Float[Tensor, "batch frame 3 3"] | None,
    ]:
        geometric_modalities = [subset for subset in modalities_out.subsets if isinstance(subset, GeometryModalities)]
        assert len(geometric_modalities) == 1
        geometric_modalities = geometric_modalities[0]
        assert geometric_modalities.parameterization in ["depth", "local_pointmap"]

        # Prepare depths and surfaces.
        geometric_predictions = geometric_modalities.cat_sequence_multiplicity(output_dict, dim=1)
        geometric_predictions = geometric_modalities.to_geometry(geometric_predictions)

        if geometric_modalities.parameterization == "local_pointmap":
            surfaces = rearrange(geometric_predictions, 'b f xyz h w -> b f h w xyz')
            depths = surfaces[...,-1]
            intrinsics = intrinsics_from_3D(surfaces)
        elif geometric_modalities.parameterization == "depth":
            surfaces = None
            intrinsics = None
            depths = geometric_predictions
        else:
            raise NotImplementedError(f'Geometry parameterization {geometric_modalities.parameterization} not supported in Diffmap.')
        
        B, F, H, W = depths.shape
        device = depths.device

        # Prepare correspondence weights.
        correspondence_weights = output_dict.get('conf', None)
        if correspondence_weights is None:
            correspondence_weights = torch.ones((B,F-1,H,W), dtype=torch.float32, device=device)

        # Prepare gt flows.
        flows = batch['flows']
        assert isinstance(flows, Flows)

        # Prepare dummy flowmap batch.
        dummy_flowmap_batch = {
            "videos": torch.zeros((B, F, 3, H, W), dtype=torch.float32, device=device),
            "indices": repeat(torch.arange(F, device=device), 'f -> b f', b=B),
            "scenes": [""]*B,
            "datasets": [""]*B,
        }
        # if 'ctxt_camera' in batch and 'trgt_camera' in batch:
        #     intrinsics_ctxt = torch.stack([cam.K for cam in batch['ctxt_camera']], dim=0).float() #(frame 3 3)
        #     intrinsics_trgt = torch.stack([cam.K for cam in batch['trgt_camera']], dim=0).float() #(frame 3 3)
        #     intrinsics = torch.stack([intrinsics_ctxt, intrinsics_trgt], dim=1).to(device) #(frame pair=1 3 3)
        #     dummy_flowmap_batch['intrinsics'] = intrinsics
        batch = Batch(**dummy_flowmap_batch)
        batch, _ = crop_and_resize_batch_for_model(batch, self.cfg_cropping)
        b, f, _, h, w = batch.videos.shape
        assert H == h and W == w

        return batch, flows, depths, correspondence_weights, surfaces, intrinsics

    def forward(
        self,
        output_dict: Dict[str, Float[Tensor, "batch frame channel height width"]],
        modalities_in: Modalities,
        modalities_out: Modalities,
        batch: Dict,
        prefix: str,
        global_step: int,
        return_outputs: bool = False,
        **kwargs
    ) -> tuple[Loss, Dict] | tuple[Loss, Dict, ModelOutput]:

        batch, flows, depths, correspondence_weights, surfaces, intrinsics = self.prepare_loss_inputs(output_dict, batch, modalities_in, modalities_out)

        # Compute poses and intrinsics using Procrustes alignment.
        model_output = self.model(batch, flows, depths, correspondence_weights, global_step, surfaces, intrinsics)

        # Compute and log the loss.
        total_loss = 0
        for loss_fn in self.losses:
            loss = loss_fn.forward(batch, flows, None, model_output, global_step)
            # self.log(f"train/loss/{loss_fn.cfg.name}", loss)
            total_loss = total_loss + loss

        # Log intrinsics error.
        if batch.intrinsics is not None:
            fx_hat = reduce(model_output.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_hat = reduce(model_output.intrinsics[..., 1, 1], "b f ->", "mean")
            fx_gt = reduce(batch.intrinsics[..., 0, 0], "b f ->", "mean")
            fy_gt = reduce(batch.intrinsics[..., 1, 1], "b f ->", "mean")
            # self.log("train/intrinsics/fx_error", (fx_gt - fx_hat).abs())
            # self.log("train/intrinsics/fy_error", (fy_gt - fy_hat).abs())

        # Logs.
        depths = model_output.depths.detach().cpu() # (B, F, H, W), depth maps
        surfaces = model_output.surfaces.detach().cpu() # (B, F, H, W, XYZ=3), local pointmaps
        intrinsics = model_output.intrinsics.detach().cpu() # (B, F, 3, 3), estimated camera intrinsics
        extrinsics = model_output.extrinsics.detach().cpu() # (B, F, 4, 4), estimated camera extrinsics

        focals = intrinsics[:,:,0,0]
        translations = extrinsics[:,:,:3,3]
        metrics_dict = {
            'flowmap/translation_norm': translations.norm(dim=-1).mean(), # TODO - adapt in case not pairwise
            'flowmap/normalized_focal': focals.mean(),
            'flowmap/normalized_focal_std': focals.std(),
            'flowmap/depth_mean': depths.mean(),
            'flowmap/depth_std': depths.std(),
            'flowmap/surfaces_norm_mean': surfaces.norm(dim=-1).mean(),
            'flowmap/surfaces_norm_std': surfaces.norm(dim=-1).std()
        }
        metrics_dict.update({f'{prefix}/flowmap_loss': total_loss})

        if return_outputs:
            return total_loss, metrics_dict, model_output
        else:
            return total_loss, metrics_dict
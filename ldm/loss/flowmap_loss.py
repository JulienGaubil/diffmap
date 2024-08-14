import torch
import torch.nn as nn

from dataclasses import dataclass
from torch import Tensor
from jaxtyping import Float
from typing import Tuple, Type, TypeVar
from einops import rearrange, reduce
from omegaconf import OmegaConf, DictConfig

from ldm.thirdp.flowmap.flowmap.model.intrinsics import IntrinsicsCfg, get_intrinsics
from ldm.thirdp.flowmap.flowmap.model.extrinsics import ExtrinsicsCfg, get_extrinsics
from ldm.thirdp.flowmap.flowmap.model.model import ModelOutput
from ldm.thirdp.flowmap.flowmap.flow.flow_predictor import Flows
from ldm.thirdp.flowmap.flowmap.dataset.types import Batch
from ldm.thirdp.flowmap.flowmap.model.backbone.backbone import BackboneOutput
from ldm.thirdp.flowmap.flowmap.model.projection import sample_image_grid, unproject
from ldm.thirdp.flowmap.flowmap.misc.cropping import CroppingCfg, crop_and_resize_batch_for_model, crop_and_resize_batch_for_flow
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
class FlowmapModelDiffCfg:
    intrinsics: IntrinsicsCfg
    extrinsics: ExtrinsicsCfg
    use_correspondence_weights: bool

@dataclass
class CommonDiffmapCfg:
    model: FlowmapModelDiffCfg
    loss: list[LossCfg]
    cropping: CroppingCfg


@dataclass
class FlowmapLossCfg:
    patch_size: int


@dataclass
class DiffmapCfg(CommonDiffmapCfg):
    model_wrapper: FlowmapLossCfg



class FlowmapModelDiff(nn.Module):
    def __init__(
        self,
        cfg: FlowmapModelDiffCfg,
        num_frames: int | None = None,
        image_shape: tuple[int, int] | None = None,
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
    ) -> ModelOutput:
        device = batch.videos.device
        _, _, _, h, w = batch.videos.shape

        # Run the backbone, which provides depths and correspondence weights TODO replace hack
        backbone_out = BackboneOutput(**{
            "depths": depths,
            "weights": correspondence_weights
            }
        )

        # Allow the correspondence weights to be ignored as an ablation.
        if not self.cfg.use_correspondence_weights:
            backbone_out.weights = torch.ones_like(backbone_out.weights)

        # Compute the intrinsics.
        intrinsics = self.intrinsics.forward(batch, flows, backbone_out, global_step)

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
        self.model = FlowmapModelDiff(cfg.model)
        self.losses = get_losses(cfg.loss)

    # @torch.no_grad()
    def preprocess_inputs(self,
            batch_dict: dict,
            flows: Flows,
            depths: Float[Tensor, "batch pair frame height width"],
        ) -> tuple[Batch, Flows, Float[Tensor, "batch frame height_scaled width_scaled"]]:
        # Convert the batch from an untyped dict to a typed dataclass.
        batch = Batch(**batch_dict)
        batch, _ = crop_and_resize_batch_for_model(batch, self.cfg_cropping)
        b, f, _, h, w = batch.videos.shape
        
        # Rescale depth
        depths = rearrange(depths[...,None], "b f h w xy -> (b f) xy h w")
        depths = rearrange(depths, "(b f) xy h w -> b f h w xy", b=b, f=f).squeeze(-1)

        return batch, flows, depths

    def forward(
            self,
            batch: dict,
            flows: Flows,
            depths: Float[Tensor, "batch frame height width"],
            correspondence_weights: Float[Tensor, "batch frame-1 height width"],
            global_step: int,
            return_outputs = False
    ) -> Loss | tuple[Loss, ModelOutput]:
        batch, flows, depths = self.preprocess_inputs(batch, flows, depths)

        # Compute depths, poses, and intrinsics using the model.
        model_output = self.model(batch, flows, depths, correspondence_weights, global_step)

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

        if return_outputs:
            return total_loss, model_output
        else:
            return total_loss

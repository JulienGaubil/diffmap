import torch
from jaxtyping import install_import_hook, Float
from torch import Tensor
import argparse

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from ..modules.flowmap.config.common import get_typed_root_config
    from ..modules.flowmap.config.pretrain import PretrainCfg
    from ..modules.flowmap.dataset.data_module_pretrain import DataModulePretrain
    from ..modules.flowmap.dataset.types import Batch
    from ..modules.flowmap.flow import FlowPredictorRaftCfg, FlowPredictorRaft, Flows, get_flow_predictor
    from ..modules.flowmap.misc.common_training_setup import run_common_training_setup
    from ..modules.flowmap.misc.cropping import (
        crop_and_resize_batch_for_flow,
        crop_and_resize_batch_for_model,
        CroppingCfg
    )
    from ..modules.flowmap.dataset.dataset_llff import load_image as load_frame_llff


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)

    parser.add_argument(
        "-c",
        "--config",
        type = str,
        default = "configs/preprocessing/preprocess_flow.yaml",
        help = "path to config."
    )

    return parser

@torch.no_grad()
def compute_flows(
    batch: Batch,
    flow_shape: tuple[int, int],
    device: torch.device,
    cfg: FlowPredictorRaftCfg | None = None,
    flow_predictor: FlowPredictorRaft | None = None, 
) -> Flows:
    # Get flow predictor.
    if flow_predictor is None:
        assert cfg is not None
        flow_predictor = get_flow_predictor(cfg)

    flow_predictor.to(device)
    return flow_predictor.compute_bidirectional_flow(batch.to(device), flow_shape)

     
def preprocess_flow(
    frames: Float[Tensor, "batch 3 height width"],
    flow_shape: tuple[int, int],
    flow_predictor: FlowPredictorRaft | None = None, 
    ) -> tuple[
        Float[Tensor, "batch 3 height width"],
        Float[Tensor, "pair height width xy=2"],
        Float[Tensor, "pair height width xy=2"],
        Float[Tensor, "pair height width"],
        Float[Tensor, "pair height width"]
    ]:

    H, W = flow_shape

     # TODO - manually implement
    cropping_cfg = CroppingCfg(**{
        "image_shape": (H, W),
        "flow_scale_multiplier": 2,
        "patch_size": 32
    })

    device = torch.device("cuda:0")

    # Create flow predictor.
    if flow_predictor is None:
        flow_cfg = FlowPredictorRaftCfg(**{
            "name": "raft",
            "num_flow_updates": 32,
            "max_batch_size": 8,
            "show_progress_bar": True
        })
        flow_predictor = get_flow_predictor(flow_cfg)
        flow_predictor.to(device)

    flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask, frames_save = list(), list(), list(), list(), list()

    # for  i, batch in enumerate(dataloader_train):
    for i in range(len(frames) - 1):
        # batch.pop("frame_paths", None)
        # TODO implement
        batch_pair = {
            "videos": frames[i:i+2].unsqueeze(0),
            "indices": torch.Tensor([i,i+1]),
            "scenes": [""],
            "datasets": "",
            "extrinsics": None,
            "intrinsics": None
        }
        # batch = Batch(**batch)
        batch = Batch(**batch_pair)
        print("Indices : ", batch.indices)
        batch_for_model, (h,w) = crop_and_resize_batch_for_model(batch, cropping_cfg)
        batch_for_flow = crop_and_resize_batch_for_flow(batch, cropping_cfg)

        # Compute optical flow.
        flows = compute_flows(batch_for_flow, (h, w), device, flow_cfg, flow_predictor)

        flow_fwd = flows.forward[0,0,:,:,:].cpu() # (H, W, 2)
        flow_bwd = flows.backward[0,0,:,:,:].cpu() # (H, W, 2)
        flow_fwd_mask = flows.forward_mask[0,0,:,:].cpu() # (H, W)
        flow_bwd_mask = flows.backward_mask[0,0,:,:].cpu() # (H, W)
        prev_frame = batch_for_model.videos[0,0,:,:,:] # (3, H, W)
        curr_frame = batch_for_model.videos[0,1,:,:,:] # (3, H, W)

        flows_fwd.append(flow_fwd)
        flows_bwd.append(flow_bwd)
        flows_fwd_mask.append(flow_fwd_mask)
        flows_bwd_mask.append(flow_bwd_mask)
        frames_save.append(prev_frame)

        if i == len(frames) - 2:
            frames_save.append(curr_frame)

    frames_save = torch.stack(frames_save, dim=0) # (B, 3, H, W)
    flows_fwd = torch.stack(flows_fwd, dim=0) # (B, H, W, 2)
    flows_bwd = torch.stack(flows_bwd, dim=0) # (B, H, W, 2)
    flows_fwd_mask = torch.stack(flows_fwd_mask, dim=0) # (B, H, W)
    flows_bwd_mask = torch.stack(flows_bwd_mask, dim=0) # (B, H, W)

    print(frames_save.size(), flows_fwd.size(), flows_bwd.size(), flows_fwd_mask.size(), flows_bwd_mask.size())

    return frames_save, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask
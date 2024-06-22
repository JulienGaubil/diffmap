import glob
import os, os.path
import torch
import hydra
import numpy as np

from pathlib import Path
from jaxtyping import install_import_hook, Float
from torch import Tensor
from omegaconf import OmegaConf, ListConfig, DictConfig
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from flow_vis_torch import flow_to_color
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
from dataclasses import replace

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from ..modules.flowmap.dataset.dataset_llff import load_image as load_frame_llff
    from ..modules.flowmap.misc.cropping import (
        crop_and_resize_batch_for_flow,
        crop_and_resize_batch_for_model,
        CroppingCfg
    )
    from ..modules.flowmap.dataset.types import Batch
    from ..modules.flowmap.flow import compute_flows

# from .preprocess import preprocess_flow, get_parser
from .preprocess_llff import load_frames_llff, dump_llff
from ..modules.flowmap.dataset.dataset_llff import load_image as load_frame_co3d
from ..modules.flowmap.flow import FlowPredictorRaftCfg, FlowPredictorRaft, Flows, get_flow_predictor


def dump_scene(
    batch,
    flows: Flows,
    scene_path: Path,
    stride: int
) -> None:
    
    if stride == 1:
        suffix = ""
    else:
        suffix = f"_stride_{stride}"

    frames = batch.videos.squeeze(0) # (frame 3 height width)
    indices = batch.indices.squeeze(0) # (frame)
    fwd_flows = flows.forward.squeeze(0) # (pair=frame-1 3 height width 2)
    bwd_flows = flows.backward.squeeze(0) # (pair=frame-1 3 height width 2)
    masks_flow_fwd = flows.forward_mask.squeeze(0) # (pair=frame-1 3 height width)
    masks_flow_bwd = flows.backward_mask.squeeze(0) # (pair=frame-1 3 height width)
    
    img_path = os.path.join(scene_path, "images_diffmap_raft" + suffix)
    flow_fwd_path = os.path.join(scene_path, "flow_forward_raft" + suffix)
    flow_bwd_path = os.path.join(scene_path, "flow_backward_raft" + suffix)

    os.makedirs(flow_fwd_path, exist_ok=True)
    os.makedirs(flow_bwd_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    assert indices.size(0) -1 == frames.size(0) - 1  == fwd_flows.size(0) == bwd_flows.size(0) == masks_flow_fwd.size(0) == masks_flow_bwd.size(0)

    for i in tqdm(range(frames.size(0)), desc="dumping files"):
        curr_idx = int(indices[i])

        if i < frames.size(0) - 1:
            # Clone slices (else saves the whole tensor, see: https://discuss.pytorch.org/t/saving-tensor-with-torch-save-uses-too-much-memory/46865/3)
            fwd_flow = fwd_flows[i].clone().to(torch.float16)
            bwd_flow = bwd_flows[i].clone().to(torch.float16)
            mask_flow_fwd = masks_flow_fwd[i].clone().to(torch.float16)
            mask_flow_bwd = masks_flow_bwd[i].clone().to(torch.float16)
            curr_frame = frames[i].clone()
            # next_frame = frames[i + 1].clone()
            next_idx = int(indices[i+1])
            
            # Save flow, RGB flow viz and frames.
            torch.save(fwd_flow, flow_fwd_path / Path(f'flow_fwd_%06d_%06d.pt'%(curr_idx, next_idx)) )
            torch.save(bwd_flow, flow_bwd_path / Path(f'flow_bwd_%06d_%06d.pt'%(curr_idx, next_idx)) )
            torch.save(mask_flow_fwd, flow_fwd_path / Path(f'mask_flow_fwd_%06d_%06d.pt'%(curr_idx, next_idx)) )
            torch.save(mask_flow_bwd, flow_bwd_path / Path(f'mask_flow_bwd_%06d_%06d.pt'%(curr_idx, next_idx)) )

            fwd_flow_viz = rearrange(fwd_flow, "h w xy -> xy h w")
            bwd_flow_viz = rearrange(bwd_flow, "h w xy -> xy h w")
            fwd_flow_viz = flow_to_color(fwd_flow_viz) / 255
            bwd_flow_viz = flow_to_color(bwd_flow_viz) / 255
            # save_image(fwd_flow_viz, flow_fwd_path / Path(f'flow_fwd_%06d_%06d.png'%(curr_idx, next_idx)) )
            # save_image(bwd_flow_viz, flow_bwd_path / Path(f'flow_bwd_%06d_%06d.png'%(curr_idx, next_idx)) )
        
        save_image(curr_frame, img_path / Path(f"frame%06d.png"%curr_idx) )


def prepare_raw_batches(scene_path: str | Path, stride: int = 1) -> list[Batch]:
    # Get rgb paths.
    rgb_file_folder_path = Path(scene_path / "images")
    all_rgb_files = [Path(p) for p in sorted(glob.glob(os.path.join(rgb_file_folder_path, "*.jpg")))]
    N = len(all_rgb_files)

    batches = list()

    print(0.001, stride)
    # Load frames.
    for s in range(stride):
        rgb_files = all_rgb_files[s::stride]
        indices = [k for k in np.arange(s, N, stride)]
        frames = list()

        print(0.01, [path.name for path in rgb_files])
        print(0.02, indices)
        
        for k in range(len(rgb_files)):
            frame = np.asarray(Image.open(rgb_files[k]).convert("RGB")) #(H, W, 3), [0, 255]
            frame = frame / 255
            frame = rearrange(torch.from_numpy(frame.copy()), 'h w c -> c h w').to(torch.float32)
            frames.append(frame) # (3, H, W), [0,1]

        frames = torch.stack(frames, dim=0).unsqueeze(0)
        indices = repeat(torch.tensor(indices), 'n -> () n')
        print(0.03, frames.shape)
        print('')

        batch = Batch(videos=frames, indices=indices, scenes=[""], datasets=[""])
        batches.append(batch)
    
    return batches


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

     
def preprocess_flow_raft(
    batch: Batch,
    flow_shape: tuple[int, int],
    flow_predictor: FlowPredictorRaft | None = None,
) -> tuple[
    Batch,
    Flows
]:
    # Flowmap cropping config
    cropping_cfg = CroppingCfg(**{
        "image_shape": 43200,
        "flow_scale_multiplier": 4,
        "patch_size": 32
    })

    device = torch.device("cuda")
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

    # Crop and resize batch for flow.
    batch_for_flow = crop_and_resize_batch_for_flow(batch, cropping_cfg)
    
    # Compute flow.
    # TODO - remove hack
    H, W = batch_for_flow.videos.shape[-2:]
    flows = compute_flows(batch_for_flow, (H, W), device, flow_cfg)

    flows = replace(flows, forward=flows.forward.cpu(), backward=flows.backward.cpu(), forward_mask=flows.forward_mask.cpu(), backward_mask=flows.backward_mask.cpu())
    batch_for_flow = replace(batch_for_flow, videos=batch_for_flow.videos.cpu(), indices=batch_for_flow.indices.cpu())

    return batch_for_flow, flows

@hydra.main(
        version_base=None,
        config_path='../../configs/preprocessing',
        config_name='preprocess_co3d'
)
def preprocess_co3d(cfg: DictConfig) -> None:
    # Saving paths.
    root = Path(cfg['data']['root'])
    scenes = cfg['data']['scenes']
    categories = cfg['data']['categories']
    stride = cfg['data']['stride']
    H, W = cfg['data']['image_shape'][0], cfg['data']['image_shape'][1]

    # Load scenes.
    if scenes is not None:
        if isinstance(scenes, (list, ListConfig)):
            scenes = sorted(list(set([str(scene) for scene in scenes])))
        elif isinstance(scenes, str) or isinstance(scenes, int):
            scenes = [str(scenes)]
        else:
            raise AssertionError(f"Scenes field must be str or list in config, got {type(scenes)}.")

    scene_paths = list()
    for category in categories:
        scenes_paths_category = sorted([path.name for path in (root / category).iterdir() if path.is_dir()])
        
        if scenes is not None:
            scene_paths += [(root / category / scene) for scene in scenes if scene in scenes_paths_category]
        else:
            scene_paths += [(root / category / scene ) for scene in scenes_paths_category]

    print(0.0, scene_paths)
    print(0.1, len(scene_paths))

    for scene_path in scene_paths:
        print(scene_path)

        # Get raw batch.
        batches = prepare_raw_batches(scene_path, stride=stride)

        for batch in batches:

            # TODO - properly handle in configs
            H, W = batch.videos.shape[-2:]

            # Compute and save flow.
            batch_for_flow, flows = preprocess_flow_raft(batch, (H, W), flow_predictor=None)
            print(0.1, batch_for_flow.videos.shape, flows.forward.shape)
            dump_scene(batch_for_flow, flows, scene_path, stride)


if __name__ == "__main__":
    preprocess_co3d()
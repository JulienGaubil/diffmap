import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'ldm/modules/flowmap/third_party/raft/core'))

import glob
import os, os.path
import numpy as np
import torch, torchvision
import hydra
import json
import argparse

from pathlib import Path
from einops import rearrange
from tqdm import tqdm
from flow_vis_torch import flow_to_color
from jaxtyping import Float
from omegaconf import OmegaConf, DictConfig, ListConfig
from torchvision.utils import save_image
from torch import Tensor
from torchvision.transforms import functional as FT

from .utils import BatchRunningStd
from ldm.data.utils.tforms import CenterCropFlow
from ldm.misc.projection import compute_consistency_mask
from ..modules.flowmap.flow import Flows
from ..modules.flowmap.third_party.raft.core.raft import RAFT
from ..modules.flowmap.third_party.raft.core.utils.utils import InputPadder, forward_interpolate

std_estimator = BatchRunningStd()

def resize_flow_mask(flow_mask, size):
    flow_mask = rearrange(flow_mask, 'h w -> () h w')
    flow_mask_resized = FT.center_crop(FT.resize(flow_mask, size), size)
    flow_mask_resized = flow_mask_resized.squeeze()
    return flow_mask_resized.to(torch.float16)


def resize_flow(flow, size):
    center_crop_flow = CenterCropFlow(size)
    flow = rearrange(flow, 'h w c -> c h w')
    flow_resized = center_crop_flow(FT.resize(flow, size))
    flow_resized = rearrange(flow_resized, 'c h w -> h w c')
    return flow_resized.to(torch.float16)

def resize_im(im, size):
    return FT.center_crop(FT.resize(im, size), size)
    

def dump_scene(
    frames: Float[Tensor, "batch 3 height width"],
    flows: Flows,
    scene_path: Path,
    stride: int,
    size: list[int]
) -> None:
    global std_estimator

    fwd_flows = flows.forward.squeeze(0) # (pair=frame-1 height width 2)
    bwd_flows = flows.backward.squeeze(0) # (pair=frame-1 height width 2)
    masks_flow_fwd = flows.forward_mask.squeeze(0) # (pair=frame-1 height width)
    masks_flow_bwd = flows.backward_mask.squeeze(0) # (pair=frame-1 height width)

    if stride == 1:
        suffix = ""
    else:
        suffix = f"_stride_{stride}"

    img_path = os.path.join(scene_path, "images_diffmap_raft" + suffix)
    flow_fwd_path = os.path.join(scene_path, "flow_forward_raft" + suffix)
    flow_bwd_path = os.path.join(scene_path, "flow_backward_raft" + suffix)

    os.makedirs(flow_fwd_path, exist_ok=True)
    os.makedirs(flow_bwd_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)
    os.system(f'rm {os.path.join(img_path, "*.jpg")}')
    os.system(f'rm {os.path.join(img_path, "*.png")}')
    os.system(f'rm {os.path.join(flow_fwd_path, "*.pt")}')
    os.system(f'rm {os.path.join(flow_bwd_path, "*.pt")}')

    
    assert frames.size(0) - stride  == fwd_flows.size(0) == bwd_flows.size(0) == masks_flow_fwd.size(0) == masks_flow_bwd.size(0)

    for i in range(len(frames)):
        frame = frames[i]
        curr_idx, next_idx = i, i + stride

        if i < len(fwd_flows):
            # Clone slices (else saves the whole tensor, see: https://discuss.pytorch.org/t/saving-tensor-with-torch-save-uses-too-much-memory/46865/3)
            fwd_flow = fwd_flows[i].clone().to(torch.float16)
            bwd_flow = bwd_flows[i].clone().to(torch.float16)
            mask_flow_fwd = masks_flow_fwd[i].clone().to(torch.float16)
            mask_flow_bwd = masks_flow_bwd[i].clone().to(torch.float16)

            if size is not None:
                fwd_flow = resize_flow(fwd_flow, size)
                bwd_flow = resize_flow(bwd_flow, size)
                mask_flow_fwd = resize_flow_mask(mask_flow_fwd, size)
                mask_flow_bwd = resize_flow_mask(mask_flow_bwd, size)

            # Update running mean and std estimate.
            std_estimator.update(fwd_flow)

            # Save flow, RGB flow viz and frames.
            torch.save(fwd_flow, os.path.join(flow_fwd_path / Path(f'flow_fwd_%06d_%06d.pt' % (curr_idx, next_idx))))
            torch.save(bwd_flow, os.path.join(flow_bwd_path / Path(f'flow_bwd_%06d_%06d.pt' % (curr_idx, next_idx))))
            torch.save(mask_flow_fwd, flow_fwd_path / Path(f'mask_flow_fwd_%06d_%06d.pt' % (curr_idx, next_idx)))
            torch.save(mask_flow_bwd, flow_bwd_path / Path(f'mask_flow_bwd_%06d_%06d.pt' % (curr_idx, next_idx)))

        if size is not None:
            frame = resize_im(frame, size)

        save_image(frame, img_path / Path(f"frame%06d.jpg"%curr_idx) )
    
    print(0.8, scene_path, 'SCENE MEAN, STD : ', fwd_flows.mean(), fwd_flows.std())
    print(0.9, scene_path, 'CURRENT MEAN, STD ESTIMATE: ', std_estimator.get_mean_std())
    print('')

def compute_flows_raft(
    scene_path: Path,
    args: argparse.Namespace,
    flow_predictor: RAFT,
    stride: int
) -> tuple[Float[Tensor, "batch 3 height width"], Flows]:
    device = torch.device('cuda')

    # For warm start.
    flow_prev_fwd, flow_prev_bwd = None, None
    frames, fwd_flows, bwd_flows, flows_masks_fwd, flows_masks_bwd= list(), list(), list(), list(), list()
    indices_frames, indices_flows = list(), list()

    with torch.no_grad():
        images = glob.glob(os.path.join(scene_path, 'images', '*.png')) + \
                 glob.glob(os.path.join(scene_path, 'images', '*.jpg'))
        
        images = sorted(images)
        N = len(images)
        
        for s in range(stride):
            image_stride = images[s::stride]
            indices_stride = [k for k in np.arange(s, N, stride)]

            for i, (imfile1, imfile2) in tqdm(enumerate(zip(image_stride[:-1], image_stride[1:]))):
                curr_idx, next_idx = indices_stride[i], indices_stride[i+1]

                # Load images.
                curr_image = torchvision.io.read_image(imfile1).float() # (B=1 C H W), [0,255]
                next_image = torchvision.io.read_image(imfile2).float()
                image1 = curr_image[None].to(device)
                image2 = next_image[None].to(device)
                padder = InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2) # (B=1 C H' W'), [0,255]

                # Compute optical flows.
                iters = 20 if (flow_prev_fwd is None and args.warm_start) else args.iters
                flow_low_fwd, fwd_flow = flow_predictor(image1, image2, iters=iters, flow_init=flow_prev_fwd, test_mode=True) #(B=1 VU H W), [-H,H]x[-W,W]
                flow_low_bwd, bwd_flow = flow_predictor(image2, image1, iters=iters, flow_init=flow_prev_bwd, test_mode=True) #(B=1 VU H W), [-H,H]x[-W,W]

                # Resize and normalize flow.
                H, W = image1.shape[-2:]
                wh = torch.tensor((W, H), dtype=torch.float32)
                fwd_flow = FT.resize(fwd_flow, curr_image.shape[-2:]).cpu().squeeze()
                bwd_flow = FT.resize(bwd_flow, curr_image.shape[-2:]).cpu().squeeze()
                fwd_flow = fwd_flow / wh[:,None,None]
                bwd_flow = bwd_flow / wh[:,None,None]

                # Compute consistency masks.
                curr_image = curr_image / 255
                next_image = next_image / 255
                fwd_flow = rearrange(fwd_flow, 'xy h w -> h w xy')
                bwd_flow = rearrange(bwd_flow, 'xy h w -> h w xy')
                fwd_mask = compute_consistency_mask(curr_image, next_image, fwd_flow)
                bwd_mask = compute_consistency_mask(next_image, curr_image, bwd_flow)
                
                if args.warm_start:
                    flow_prev_fwd = forward_interpolate(flow_low_fwd[0])[None].cuda()
                    flow_prev_bwd = forward_interpolate(flow_low_bwd[0])[None].cuda()


                if i == 0:
                    frames.append(curr_image)
                    indices_frames.append(curr_idx)
                indices_frames.append(next_idx)
                frames.append(next_image)

                indices_flows.append(curr_idx)
                fwd_flows.append(fwd_flow)
                bwd_flows.append(bwd_flow)
                flows_masks_fwd.append(fwd_mask)
                flows_masks_bwd.append(bwd_mask)

        # Re-order.   
        indices_flows = torch.tensor(indices_flows)
        indices_frames = torch.tensor(indices_frames)
        perm_flows = torch.argsort(indices_flows)
        perm_frames = torch.argsort(indices_frames)

        frames = torch.stack([frames[id] for id in perm_frames], dim=0) # (frame 3 height width)
        fwd_flows = torch.stack([fwd_flows[id][None] for id in perm_flows], dim=1) # (batch=1 pair=frame-stride height width 2)
        bwd_flows = torch.stack([bwd_flows[id][None] for id in perm_flows], dim=1) # (batch=1 pair=frame-stride height width 2)
        flows_masks_fwd = torch.stack([flows_masks_fwd[id][None] for id in perm_flows], dim=1) # (batch=1 pair=frame-stride height width)
        flows_masks_bwd = torch.stack([flows_masks_bwd[id][None] for id in perm_flows], dim=1) # (batch=1 pair=frame-stride height width)

        flows = Flows(forward=fwd_flows, backward=bwd_flows, forward_mask=flows_masks_fwd, backward_mask=flows_masks_bwd)

        print(0.5, frames.shape, fwd_flows.shape)
    return frames, flows


@hydra.main(
        version_base=None,
        config_path='../../configs/preprocessing',
        config_name='preprocess_co3d'
)
def preprocess_co3d_raft(cfg: DictConfig) -> None:
    global std_estimator

    # Saving paths.
    root = Path(cfg['data']['root'])
    scenes = cfg['data']['scenes']
    categories = cfg['data']['categories']
    stride = cfg['data']['stride']
    size = cfg['data']['image_shape']

    # Load flow predictor.
    device = torch.device('cuda')
    args = argparse.Namespace()
    args.__dict__ = OmegaConf.to_container(cfg.raft)
    flow_predictor = torch.nn.DataParallel(RAFT(args))
    flow_predictor.load_state_dict(torch.load(args.model))
    flow_predictor = flow_predictor.module
    flow_predictor.to(device)
    flow_predictor.eval()

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
        # print(0.4, scenes_paths_category)
        if scenes is not None:
            scene_paths += [(root / category / scene) for scene in scenes if scene in scenes_paths_category]
        else:
            scene_paths += [(root / category / scene) for scene in scenes_paths_category]

    print(0.01, scene_paths)
    print('')
    print(0.1, stride)

    for scene_path in scene_paths:
        print(scene_path)

        # Compute and save flow.
        frames, flows = compute_flows_raft(scene_path, args, flow_predictor, stride)
        dump_scene(frames, flows, scene_path, stride, size)

    print(0.10, 'FINAL STD, MEAN ESTIMATE: ', std_estimator.get_mean_std())
 
if __name__ == "__main__":
    preprocess_co3d_raft()


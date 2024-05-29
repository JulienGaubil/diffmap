import glob
import os, os.path
import numpy as np
import cv2
import torch, torchvision
import hydra

from pathlib import Path
from einops import rearrange
from tqdm import tqdm
from flow_vis_torch import flow_to_color
from jaxtyping import Float
from omegaconf import OmegaConf, DictConfig, ListConfig
from torchvision.utils import save_image
from torch import Tensor
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

from ldm.data.utils.camera import Camera, Intrinsics, Extrinsics
from ldm.data.utils.io import load_exr
from ldm.data.utils.camera import pixel_grid_coordinates
from ldm.misc.projection import compute_flow_projection, compute_consistency_mask
from ldm.modules.flowmap.model.projection import sample_image_grid


def dump_rooms(
    frames: Float[Tensor, "batch 3 height width"],
    fwd_flows: Float[Tensor, "pair height width xy=2"],
    bwd_flows: Float[Tensor, "pair height width xy=2"],
    masks_flow_fwd: Float[Tensor, "pair height width"],
    masks_flow_bwd: Float[Tensor, "pair height width"],
    scene_path: Path,
    stride: int
) -> None:
    
    if stride == 1:
        suffix = ""
    else:
        suffix = f"_stride_{stride}"

    img_path = os.path.join(scene_path, "images_diffmap_projection" + suffix)
    flow_fwd_path = os.path.join(scene_path, "flow_forward_projection" + suffix)
    flow_bwd_path = os.path.join(scene_path, "flow_backward_projection" + suffix)

    os.makedirs(flow_fwd_path, exist_ok=True)
    os.makedirs(flow_bwd_path, exist_ok=True)
    os.makedirs(img_path, exist_ok=True)

    assert frames.size(0) - 1 == fwd_flows.size(0) == bwd_flows.size(0) == masks_flow_fwd.size(0) == masks_flow_bwd.size(0)

    for i in tqdm(range(fwd_flows.size(0)), desc="dumping files"):
        # Clone slices (else saves the whole tensor, see: https://discuss.pytorch.org/t/saving-tensor-with-torch-save-uses-too-much-memory/46865/3)
        fwd_flow = fwd_flows[i].clone()
        bwd_flow = bwd_flows[i].clone()
        mask_flow_fwd = masks_flow_fwd[i].clone()
        mask_flow_bwd = masks_flow_bwd[i].clone()
        curr_frame = frames[i].clone()
        next_frame = frames[i+1].clone()
        
        # Save flow, RGB flow viz and frames.
        torch.save(fwd_flow, flow_fwd_path / Path(f'flow_fwd_%06d_%06d.pt'%(i,i+1)) )
        torch.save(bwd_flow, flow_bwd_path / Path(f'flow_bwd_%06d_%06d.pt'%(i,i+1)) )
        torch.save(mask_flow_fwd, flow_fwd_path / Path(f'mask_flow_fwd_%06d_%06d.pt'%(i,i+1)) )
        torch.save(mask_flow_bwd, flow_bwd_path / Path(f'mask_flow_bwd_%06d_%06d.pt'%(i,i+1)) )

        fwd_flow_viz = rearrange(fwd_flow, "h w xy -> xy h w")
        bwd_flow_viz = rearrange(bwd_flow, "h w xy -> xy h w")
        fwd_flow_viz = flow_to_color(fwd_flow_viz) / 255
        bwd_flow_viz = flow_to_color(bwd_flow_viz) / 255
        # save_image(fwd_flow_viz, flow_fwd_path / Path(f'flow_fwd_%06d_%06d.png'%(i,i+1)) )
        # save_image(bwd_flow_viz, flow_bwd_path / Path(f'flow_bwd_%06d_%06d.png'%(i+1,i)) )
        save_image(curr_frame, img_path / Path(f"frame%06d.png"%i) )

        # Save last frame.
        if i == fwd_flows.size(0) - 1:
            save_image(next_frame, img_path / Path(f"frame%06d.png"%(i+1)) )

def preprocess_flow_projection(
    scene_path: Path,
    resolution: list[int,int],
    stride: int = 1
) -> tuple[
    Float[Tensor, "batch 3 height width"],
    Float[Tensor, "pair height width xy=2"],
    Float[Tensor, "pair height width xy=2"],
    Float[Tensor, "pair height width"],
    Float[Tensor, "pair height width"]
]:
    H_trgt, W_trgt = resolution

    # Get depth paths.
    depth_file_folder_path = scene_path / "depth_exr"
    depth_files = [Path(p) for p in sorted(glob.glob(os.path.join(depth_file_folder_path, f"*.exr")))]
    # depth_files = depth_files[::stride]

    # Get rgb paths.
    rgb_file_folder_path = Path(scene_path / "rgb")
    rgb_files = [Path(p) for p in sorted(glob.glob(os.path.join(rgb_file_folder_path, "*.png")))]
    # rgb_files = rgb_files[::stride]

    # Load intrinsics.
    intrinsics_file = scene_path / "intrinsics.txt"
    K = np.loadtxt(intrinsics_file, max_rows=1)
    fx = fy = K[0]
    cx = K[1]
    cy = K[2]
    skew = 0
    HW = np.loadtxt(intrinsics_file, skiprows=3).astype(int)
    H, W = HW[0], HW[1]

    # Rescale intrinsics.
    ratio_H, ratio_W = H / H_trgt, W / W_trgt
    fx = fx / ratio_W
    cx = cx / ratio_W
    fy = fy / ratio_H
    cy = cy / ratio_H

    # Create intrinsics.
    intrinsics = Intrinsics(**{
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "skew": skew,
        "resolution": [H_trgt,W_trgt]
    })

    # Load extrinsics.
    extrinsics_file = scene_path / "poses.txt"
    c2ws = np.loadtxt(extrinsics_file)
    c2ws = torch.from_numpy(c2ws.reshape((-1,4,4)))
    w2cs = torch.linalg.inv(c2ws)
    # w2cs = w2cs[::stride]

    # Create cameras.
    cameras = [
        Camera(
            extrinsics=Extrinsics(R=w2cs[k,:3,:3], t=w2cs[k,:3,3]),
            intrinsics=intrinsics
        )
        for k in range(w2cs.shape[0])
    ]
    
    fwd_flows, bwd_flows = list(), list()
    fwd_masks, bwd_masks = list(), list()
    frames = list()
    for idx in tqdm(range(0, len(depth_files) - 1, stride)):

        prev_camera = cameras[idx]
        curr_camera = cameras[idx + stride]

        # Load and resize depth maps.
        depth_pixels_prev = load_exr(depth_files[idx])
        depth_pixels_curr = load_exr(depth_files[idx + stride])
        depth_pixels_prev = transforms.functional.resize(
            depth_pixels_prev, 
            [H_trgt, W_trgt],
            interpolation=transforms.InterpolationMode.NEAREST
        ).type(torch.float) #(3, H, W), [0, 255]
        depth_pixels_curr = transforms.functional.resize(
            depth_pixels_curr, 
            [H_trgt, W_trgt],
            interpolation=transforms.InterpolationMode.NEAREST
        ).type(torch.float) #(3, H, W), [0, 255]

        # Load and resize RGB frames.
        src_image = np.asarray(Image.open(rgb_files[idx]).convert("RGB")) #(H, W, 3), [0, 255]
        trgt_image = np.asarray(Image.open(rgb_files[idx+1]).convert("RGB")) #(H, W, 3), [0, 255]
        src_image = rearrange(torch.from_numpy(src_image.copy()), 'h w c -> c h w')
        trgt_image = rearrange(torch.from_numpy(trgt_image.copy()), 'h w c -> c h w')
        src_image = torchvision.transforms.functional.resize(src_image, [H_trgt, W_trgt], interpolation=transforms.InterpolationMode.BILINEAR).type(torch.float) #(3, H, W), [0, 255]
        trgt_image = torchvision.transforms.functional.resize(trgt_image, [H_trgt, W_trgt], interpolation=transforms.InterpolationMode.BILINEAR).type(torch.float) #(3, H, W), [0, 255]
        src_image = src_image / 255
        trgt_image = trgt_image / 255
        
        # Compute flows.
        fwd_flow = compute_flow_projection(
            src_depth_map=depth_pixels_prev,
            src_camera=prev_camera,
            trgt_camera=curr_camera
        )
        bwd_flow = compute_flow_projection(
            src_depth_map=depth_pixels_curr,
            src_camera=curr_camera,
            trgt_camera=prev_camera,
        )

        # Compute consistency masks.
        fwd_mask = compute_consistency_mask(src_image, trgt_image, fwd_flow)
        bwd_mask = compute_consistency_mask(trgt_image, src_image, bwd_flow)

        fwd_flows.append(fwd_flow)
        bwd_flows.append(bwd_flow)
        fwd_masks.append(fwd_mask)
        bwd_masks.append(bwd_mask)
        frames.append(src_image)

    
    frames.append(trgt_image)

    # Concatenate everything
    frames = torch.stack(frames, dim=0) # (frame, 3, h, w)
    fwd_flows = torch.stack(fwd_flows, dim=0) # (frame, h, w, vu=2)
    bwd_flows = torch.stack(bwd_flows, dim=0) # (frame, h, w, vu=2)
    fwd_masks = torch.stack(fwd_masks, dim=0) # (frame, h, w)
    bwd_masks = torch.stack(bwd_masks, dim=0) # (frame, h, w)

    return frames, fwd_flows, bwd_flows, fwd_masks, bwd_masks

@hydra.main(
        version_base=None,
        config_path='../../configs/preprocessing',
        config_name='preprocess_rooms'
)
def preprocess_rooms_projection(cfg: DictConfig) -> None:
    # Saving paths.
    root = Path(cfg["data"]["root"])
    scenes = cfg["data"]["scenes"]
    stride = cfg["data"]["stride"]

    H, W = cfg['data']["image_shape"][0], cfg["data"]["image_shape"][1]

    # Load scenes.
    if scenes is not None:
        if isinstance(scenes, (list, ListConfig)):
            scenes = sorted(list(set([str(scene) for scene in scenes])))
        elif isinstance(scenes, str) or isinstance(scenes, int):
            scenes = [str(scenes)]
        else:
          raise AssertionError(f"Scenes field must be str or list in config, got {type(scenes)}.")
    else:
        scenes = sorted([path.name for path in root.iterdir() if path.is_dir()])

    print(scenes)
    print('')
    print(stride)

    for scene in scenes:
        print(scene)
        scene_path = root / scene

        # Compute and save flow.
        frames, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask = preprocess_flow_projection(scene_path, (H, W), stride=stride)
        dump_rooms(frames, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask, scene_path, stride)
 
if __name__ == "__main__":
    preprocess_rooms_projection()


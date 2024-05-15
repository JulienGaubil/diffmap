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
from ldm.modules.flowmap.model.projection import sample_image_grid


def dump_rooms(
        frames: Float[Tensor, "batch 3 height width"],
        fwd_flows: Float[Tensor, "pair height width xy=2"],
        bwd_flows: Float[Tensor, "pair height width xy=2"],
        masks_flow_fwd: Float[Tensor, "pair height width"],
        masks_flow_bwd: Float[Tensor, "pair height width"],
        scene_path: Path,
    ) -> None:

    img_path = scene_path / "images_diffmap_projection"
    flow_fwd_path = scene_path / "flow_forward_projection"
    flow_bwd_path = scene_path / "flow_backward_projection"

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

def compute_flows_projection(
        depth_map: Float[Tensor, "height width"],
        prev_camera: Camera,
        curr_camera: Camera,
) -> Float[Tensor, "height width vu=2"]:
    """Compute flow by un-projecting and reprojecting depth in next frame.
    """
    H_trgt, W_trgt = depth_map.size()
    HW = pixel_grid_coordinates(H_trgt, W_trgt)
    pixel_coordinates = rearrange(HW, 'h w c -> c (h w)')
    depths = rearrange(depth_map, 'h w -> 1 (h w)')

    # Un-project depths to world and project back in next frame.
    world_pts = prev_camera.pixel_to_world(pixel_coordinates, depths=depths)
    pixel_coordinates_warped, _ = curr_camera.world_to_pixel(world_pts)

    # Compute flows.
    HW_warped = rearrange(pixel_coordinates_warped, 'c (h w) -> h w c', h=H_trgt, w=W_trgt)
    flow = (HW_warped - HW).to(torch.float32)

    # Normalize flow.
    hw = torch.tensor((H_trgt, W_trgt), dtype=torch.float32)
    flow = flow / hw[None,None,:]

    # Swap to flow indexing uv -> vu to match flowmap
    flow = flow[:,:,[1,0]]

    return flow

def compute_flows_projection(
        depth_map: Float[Tensor, "height width"],
        prev_camera: Camera,
        curr_camera: Camera,
) -> Float[Tensor, "height width vu=2"]:
    """Compute flow by un-projecting and reprojecting depth in next frame.
    """
    H_trgt, W_trgt = depth_map.size()
    HW = pixel_grid_coordinates(H_trgt, W_trgt)
    pixel_coordinates = rearrange(HW, 'h w c -> c (h w)')
    depths = rearrange(depth_map, 'h w -> 1 (h w)')

    # Un-project depths to world and project back in next frame.
    world_pts = prev_camera.pixel_to_world(pixel_coordinates, depths=depths)
    pixel_coordinates_warped, _ = curr_camera.world_to_pixel(world_pts)

    # Compute flows.
    HW_warped = rearrange(pixel_coordinates_warped, 'c (h w) -> h w c', h=H_trgt, w=W_trgt)
    flow = (HW_warped - HW).to(torch.float32)

    # Normalize flow.
    hw = torch.tensor((H_trgt, W_trgt), dtype=torch.float32)
    flow = flow / hw[None,None,:]

    # Swap to flow indexing uv -> vu to match flowmap
    flow = flow[:,:,[1,0]]

    return flow

def compute_consistency_mask(
    src_frame: Float[Tensor, "3 height width"],
    trgt_frame: Float[Tensor, "3 height width"],
    flow: Float[Tensor, "height width vu=2"],
) -> Float[Tensor, "height width"]:
    
    # My way
    _, H, W = src_frame.shape
    # warped_image = warp_image_flow(src_frame, fwd_flow)
    # deltas = (source - warped_image).abs().max(dim=1).values

    # David's way
    b = 1
    f = 2
    flow = flow[None,None,:,:,:]
    src_frames = src_frame[None,:,:,:]
    trgt_frames = trgt_frame[None,:,:,:]
    source_xy, _ = sample_image_grid((H, W))
    target_xy = source_xy + rearrange(flow, "b f h w xy -> (b f) h w xy")

    # Gets target pixel for every source pixel warped with flow.
    target_pixels = F.grid_sample(
        trgt_frames,
        target_xy * 2 - 1,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ) #((bf) c h w)

    # Map pixel color differences to mask weights.
    deltas = (src_frames - target_pixels).abs().max(dim=1).values #((bf) h w)
    mask = rearrange((1 - deltas) ** 8, "(b f) h w -> b f h w", b=b, f=f - 1)
    return mask.squeeze()


def preprocess_flow_projection(
    scene_path: Path,
    resolution: list[int,int]
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

    # Get rgb paths.
    rgb_file_folder_path = Path(scene_path / "rgb")
    rgb_files = [Path(p) for p in sorted(glob.glob(os.path.join(rgb_file_folder_path, "*.png")))]

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
    for idx in tqdm(range(0, len(depth_files) - 1, 1)):

        prev_camera = cameras[idx]
        curr_camera = cameras[idx+1]

        # Load and resize depth maps.
        depth_pixels_prev = load_exr(depth_files[idx])
        depth_pixels_curr = load_exr(depth_files[idx+1])
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
        fwd_flow = compute_flows_projection(
            depth_map=depth_pixels_prev,
            prev_camera=prev_camera,
            curr_camera=curr_camera
        )
        bwd_flow = compute_flows_projection(
            depth_map=depth_pixels_curr,
            prev_camera=curr_camera,
            curr_camera=prev_camera,
        )

        # Compute consistency masks.
        fwd_mask = compute_consistency_mask(src_image, trgt_image, fwd_flow)
        bwd_mask = compute_consistency_mask(trgt_image, src_image, bwd_flow)

        fwd_flows.append(fwd_flow)
        bwd_flows.append(bwd_flow)
        fwd_masks.append(fwd_mask)
        bwd_masks.append(bwd_mask)
        frames.append(src_image)
        if idx == len(depth_files) - 2:
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

    for scene in scenes:
        print(scene)
        scene_path = root / scene

        # Compute and save flow.
        frames, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask = preprocess_flow_projection(scene_path, (H, W))
        dump_rooms(frames, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask, scene_path)
 
if __name__ == "__main__":
    preprocess_rooms_projection()


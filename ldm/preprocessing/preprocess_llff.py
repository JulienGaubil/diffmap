import torch
from pathlib import Path
from jaxtyping import install_import_hook
from omegaconf import OmegaConf
from einops import rearrange
from flow_vis_torch import flow_to_color
from torchvision.utils import save_image
import os, os.path
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

# Configure beartype and jaxtyping.
with install_import_hook(
    ("flowmap",),
    ("beartype", "beartype"),
):
    from ..modules.flowmap.dataset.dataset_llff import load_image as load_frame_llff

from .preprocess import preprocess_flow, get_parser


def load_frames_llff(
        scene_path: Path,
        image_path: Path | None = None
    ) -> Float[Tensor, "frame channel height width"]:
    if image_path is not None:
        paths = sorted(image_path.iterdir())
    else:
        paths = sorted((scene_path / "images").iterdir())

    assert len(paths) > 0, f"No image found in folder {scene_path}"
    torch_images = []
    for im_path in paths:
        im, _ = load_frame_llff(im_path, None)
        torch_images.append(im)
    return torch.stack(torch_images, dim=0)


def dump_llff(
        frames: Float[Tensor, "batch 3 height width"],
        fwd_flows: Float[Tensor, "pair height width xy=2"],
        bwd_flows: Float[Tensor, "pair height width xy=2"],
        masks_flow_fwd: Float[Tensor, "pair height width"],
        masks_flow_bwd: Float[Tensor, "pair height width"],
        scene_path: Path,
    ) -> None:

    img_path = scene_path / "images_diffmap"
    flow_fwd_path = scene_path / "flow_forward"
    flow_bwd_path = scene_path / "flow_backward"

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

     
if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    cfg = OmegaConf.load(opt.config)
    cfg = OmegaConf.to_object(cfg)

    # Saving paths.
    dataset = cfg["data"]["name"]
    root = Path(cfg["data"]["root"])
    scenes = cfg["data"]["scenes"]

    H, W = cfg['data']["image_shape"][0], cfg["data"]["image_shape"][1]

    # Load scense.
    if scenes is not None:
        if isinstance(scenes, list):
            scenes = sorted([str(scene) for scene in scenes])
        elif isinstance(scenes, str) or isinstance(scenes, int):
            scenes = [str(scenes)]
        else:
          raise AssertionError(f"Scenes field must be str or list in config, got {type(scenes)}.")
    else:
        scenes = sorted([path.name for path in root.iterdir() if path.is_dir()])

    for scene in scenes:
        print(scene)
        scene_path = root / scene
        image_path = scene_path / cfg["data"]["image_folder"] if cfg["data"]["image_folder"] is not None else None

        # Compute and save flow.
        frames = load_frames_llff(scene_path, image_path)
        frames, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask = preprocess_flow(frames, (H, W))
        dump_llff(frames, flows_fwd, flows_bwd, flows_fwd_mask, flows_bwd_mask, scene_path)
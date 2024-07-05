import glob
import torch
import os, os.path
import numpy as np

from pathlib import Path
from jaxtyping import Float, Int
from torch import Tensor
from numpy import ndarray
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, IterableDataset
from omegaconf import ListConfig

from .diffmap import DiffmapDataset

class CO3DDiffmapDataset(DiffmapDataset, Dataset):
    def __init__(self,
        root_dir: str,
        image_transforms: list = [],
        split: str = 'train',
        scenes: list | ListConfig | str | int | None = None,
        categories: list | ListConfig | str | None = None,
        val_scenes: list | ListConfig | str | int | None = None,
        stride: int = 1,
        n_future: int = 1,
        n_ctxt: int = 1,
        flip_trajectories: bool = False
    ) -> None:

        DiffmapDataset.__init__(self, root_dir, image_transforms, split)
        IterableDataset.__init__(self)

        self.root_dir = Path(root_dir)
        self.split = split
        self.stride = stride
        self.n_future = n_future
        self.n_ctxt = n_ctxt
        self.flip_trajectories = flip_trajectories

        # Load categories.
        if categories is not None:
            self.categories = self.load_list_config(categories)
        else:
            self.categories = sorted([path.name for path in self.root_dir.iterdir() if path.is_dir()])

        # Load val scenes - default no validation.
        val_scenes = self.load_list_config(val_scenes)
        val_scenes = self.load_scenes(val_scenes, default_all=False)
        if self.split == 'validation':
            self.scenes = val_scenes

        # Load train scenes - default all scenes.
        elif split == 'train':
            scenes = self.load_list_config(scenes)
            self.scenes = self.load_scenes(scenes, default_all=True)

            # Remove val scenes.
            for scene in val_scenes:
                if scene in self.scenes:
                    self.scenes.remove(scene)
            assert len(self.scenes) > 0, "No train scene provided."
        
        print("Scenes : ", self.scenes)
        print('')
        
        # Prepare pairs of frames and flows.
        self.frame_pair_paths, self.flow_fwd_paths, self.flow_bwd_paths, self.flow_fwd_mask_paths, self.flow_bwd_mask_paths = list(), list(), list(), list(), list()
        for k in range(len(self.scenes)):
            scene = self.scenes[k]

            # Load flow and image paths.
            if self.stride == 1:
                paths_frames = sorted((self.root_dir / scene / "images_diffmap_raft").iterdir())
                paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_raft", "flow_fwd_*.pt")))
                paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_raft", "flow_bwd_*.pt")))
                paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_raft", "mask_flow_fwd*.pt")))
                paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_raft", "mask_flow_bwd*.pt")))
            # TODO - remove hack
            elif self.stride == 3:
                paths_frames = sorted((self.root_dir / scene / "images_diffmap_raft_stride_3").iterdir())
                paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_raft_stride_3", "flow_fwd_*.pt")))
                paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_raft_stride_3", "flow_bwd_*.pt")))
                paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_raft_stride_3", "mask_flow_fwd*.pt")))
                paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_raft_stride_3", "mask_flow_bwd*.pt")))
            # TODO - remove hack
            elif self.stride == 12:
                paths_frames = sorted((self.root_dir / scene / "images_diffmap_raft_stride_12").iterdir())
                paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_raft_stride_12", "flow_fwd_*.pt")))
                paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_raft_stride_12", "flow_bwd_*.pt")))
                paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_raft_stride_12", "mask_flow_fwd*.pt")))
                paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_raft_stride_12", "mask_flow_bwd*.pt")))
            else:
                raise Exception(f'Stride {self.stride} not valid, should be 1 or 3.')
            
            # Define dataset pairs.
            for idx in range(len(paths_frames) - ((self.n_ctxt + self.n_future - 1) * self.stride)):
                self.frame_pair_paths.append([Path(p) for p in paths_frames[idx : idx + (self.n_ctxt + self.n_future) * self.stride : self.stride]])
                self.flow_fwd_paths.append([Path(p) for p in paths_flow_fwd[idx : idx + (self.n_ctxt - 1 + self.n_future) * self.stride : stride]])
                self.flow_bwd_paths.append([Path(p) for p in paths_flow_bwd[idx : idx + (self.n_ctxt - 1 + self.n_future) * self.stride : stride]])
                self.flow_fwd_mask_paths.append([Path(p) for p in paths_flow_fwd_mask[idx : idx + (self.n_ctxt - 1 + self.n_future) * self.stride : stride]])
                self.flow_bwd_mask_paths.append([Path(p) for p in paths_flow_bwd_mask[idx : idx + (self.n_ctxt - 1 + self.n_future) * self.stride : stride]])

            assert len(self.frame_pair_paths) == len(self.flow_fwd_paths) == len(self.flow_bwd_paths) == len(self.flow_fwd_mask_paths) == len(self.flow_bwd_mask_paths)

    def load_list_config(self, raw_list_scenes: list | ListConfig | str | int | None) -> list[str]:
        if isinstance(raw_list_scenes, (list, ListConfig)):
            list_cfg = sorted(list(set([str(scene) for scene in raw_list_scenes])))
        elif isinstance(raw_list_scenes, str) or isinstance(raw_list_scenes, int):
            list_cfg = [str(raw_list_scenes)]
        elif raw_list_scenes is None:
            list_cfg = []
        else:
            raise AssertionError(f"Scenes / category field must be str, list or None in config, got {type(raw_list_scenes)}.")
        return list_cfg
    
    def load_scenes(self, scenes_list: list, default_all: bool = True) -> list[Path]:
        scenes = list()
        for category in self.categories:
            scenes_category = sorted([path.name for path in (self.root_dir / category).iterdir() if (path.is_dir() and ord(str(path.name)[0]) < 58) ])
            if len(scenes_list) > 0:
                scenes += [os.path.join(category, scene) for scene in scenes_list if scene in scenes_category]
            elif default_all:
                scenes += [os.path.join(category, scene) for scene in scenes_category]
        return scenes
    
    def _get_im(self, filename: Path) -> Float[Tensor, "height width 3"]:
        im = Image.open(filename).convert("RGB")
        return self._preprocess_im(im)
    
    def _get_flow(self, flow_paths: list[Path], flow_mask_paths: list[Path]) -> tuple[Float[Tensor, "n_future height width 3"], Float[Tensor, "n_future height width"]]:
        flows = list()
        flow_masks = list()
        for k in range(len(flow_paths)):
            # Load flow and mask.
            flow_raw = torch.load(flow_paths[k]).to(torch.float32) #(H,W,C=2)
            mask_flow_raw = torch.load(flow_mask_paths[k]).to(torch.float32)
            
            # Apply transformations.
            flow, flow_mask = self._preprocess_flow(flow_raw, mask_flow_raw)
            flows.append(flow)
            flow_masks.append(flow_mask)
        
        flows = torch.stack(flows, dim=0)
        flow_masks = torch.stack(flow_masks, dim=0)
        return flows, flow_masks
    
    def __len__(self) -> int:
        return len(self.frame_pair_paths)

    def __getitem__(self, index: int) -> dict[Float[Tensor, "..."]]:

        if self.flip_trajectories:
            pair_idx = np.random.permutation(2)
        else:
            pair_idx = np.arange(2)

        # Define paths and indices.
        if pair_idx[0] == 0:
            prev_im_paths = self.frame_pair_paths[index][:self.n_ctxt]
            future_im_paths = self.frame_pair_paths[index][self.n_ctxt:]
            fwd_flow_paths = self.flow_fwd_paths[index][-self.n_future:]
            bwd_flow_paths = self.flow_bwd_paths[index][-self.n_future:]
            fwd_flow_mask_paths = self.flow_fwd_mask_paths[index][-self.n_future:]
            bwd_flow_mask_paths = self.flow_bwd_mask_paths[index][-self.n_future:]
            indices = torch.arange(index + (self.n_ctxt - 1) * self.stride, index + (self.n_ctxt + self.n_future) * self.stride, self.stride)
        else:
            # Flipping trajectory
            prev_im_paths = self.frame_pair_paths[index][::-1][:self.n_ctxt]
            future_im_paths = self.frame_pair_paths[index][::-1][self.n_ctxt:]
            fwd_flow_paths = self.flow_bwd_paths[index][::-1][-self.n_future:]
            bwd_flow_paths = self.flow_fwd_paths[index][::-1][-self.n_future:]
            fwd_flow_mask_paths = self.flow_bwd_mask_paths[index][::-1][-self.n_future:]
            bwd_flow_mask_paths = self.flow_fwd_mask_paths[index][::-1][-self.n_future:]
            
            indices = torch.arange(index + self.n_future * self.stride, index - 1, -self.stride)

        data = {}

        # Load target, context frames.
        ctxt_ims = torch.stack([self._get_im(p) for p in prev_im_paths], dim=0)
        trgt_ims = torch.stack([self._get_im(p) for p in future_im_paths], dim=0)
        data.update({
            'ctxt_rgb': ctxt_ims,
            'trgt_rgb': trgt_ims,
            'indices': indices
            }
        )

        # Load flow
        flows_fwd, flow_fwd_masks = self._get_flow(fwd_flow_paths, fwd_flow_mask_paths)
        flows_bwd, flow_bwd_masks = self._get_flow(bwd_flow_paths, bwd_flow_mask_paths)
        data.update({
            'fwd_flow': flows_fwd,
            'bwd_flow': flows_bwd,
            'mask_fwd_flow': flow_fwd_masks,
            'mask_bwd_flow': flow_bwd_masks
            }
        )

        return data



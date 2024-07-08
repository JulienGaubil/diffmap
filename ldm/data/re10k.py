"""
Adapted from 
https://github.com/dcharatan/pixelsplat/blob/main/src/dataset/dataset_re10k.py
"""

import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path

import torch
import torchvision.transforms as tf
from einops import rearrange
from jaxtyping import Float, UInt8, Int
from PIL import Image
from torch import Tensor
from numpy import ndarray
from torch.utils.data import IterableDataset
from torchvision import transforms
from functools import partial

from .diffmap import DiffmapDataset
import numpy as np
import glob
import os, os.path


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
from typing import Literal

from .diffmap import DiffmapDataset

to_tensor = tf.ToTensor()

class Re10kDiffmapDataset(DiffmapDataset, Dataset):
    def __init__(self,
        root_dir: str,
        image_transforms: list = [],
        split: Literal["train", "validation", "test"] = "train",
        scenes: list | ListConfig | str | int | None = None,
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

        # Load val scenes - default no validation.
        if self.split == "validation":
            scenes = self.load_list_config(val_scenes)
        elif self.split == "train":
            scenes = self.load_list_config(scenes)
        self.scenes = self.load_scenes(scenes, default_all=True)
        assert len(self.scenes) > 0, "No train scene provided."
        
        print("Scenes : ", self.scenes)
        print(0.1, len(self.scenes))
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
        scenes_split = sorted([path.name for path in (self.root_dir / self.split).iterdir() if path.is_dir() ])
        if len(scenes_list) > 0:
            scenes += [os.path.join(self.split, scene) for scene in scenes_list if scene in scenes_split]
        elif default_all:
            scenes += [os.path.join(self.split, scene) for scene in scenes_split]
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










# class Re10kDiffmapDataset(DiffmapDataset, IterableDataset):
#     """
#     Iterable Re10k Diffmap dataset class.
#     """

#     def __init__(
#         self,
#         root_dir: str,
#         image_transforms: list = [],
#         trgt_key: str = "trgt",
#         ctxt_key: str = "ctxt",
#         split: str = "train",
#         scenes: list[str] | str | None = None,
#         n_val_samples_scene: int = 1
#     ) -> None:
#         DiffmapDataset.__init__(self, root_dir, image_transforms, trgt_key, ctxt_key, split)
#         IterableDataset.__init__(self)

#         # Define dataset scenes, chunk paths and samples.

#         # for root in cfg.roots:
#         #     root = root / self.data_stage
#         #     root_chunks = sorted(
#         #         [path for path in root.iterdir() if path.suffix == ".torch"]
#         #     )
#         #     self.scene_chunks.extend(root_chunks)

      

#         # if self.cfg.overfit_to_scene is not None:
#         #     chunk_path = self.index[self.cfg.overfit_to_scene]
#         #     self.scene_chunks = [chunk_path] * len(self.scene_chunks)

#         # Define scenes and associated chunk paths
#         if scenes is None:
#             raise Exception('None value for scene param of dataset not yet implemented - use str or list[str].')
#             # TODO - modify path and don't enforce a single train sequence set for generalizable setting
#             self.scene_chunks = sorted(
#                 [path for path in (self.root_dir / "train").iterdir() if path.suffix == ".torch"]
#             )
#             # self.scenes =  [path.name for path in self.root_dir.iterdir() if path.is_dir()]
#         else:
#             if isinstance(scenes, str):
#                 self.scenes = [scenes]
#                 # chunk_path = self.index[scenes]
#                 # self.scene_chunks = [chunk_path]
#             else:
#                 self.scenes = [str(scene) for scene in scenes]
#                 # self.scene_chunks = [self.index[self.cfg.overfit_to_scene]]
#             try:
#                 self.scene_chunks = sorted(list({self.index[scene] for scene in self.scenes}))
#                 print('SCENE CHUNKS : ', self.scene_chunks)
#             except KeyError:
#                 raise Exception(f'Dataset scene values should match Re10k scene keys, got unexpected value.')


#         # self.prepare_pairs(n_val_samples_scene)
#         # if self.split == "train":
#         #     self.shuffle()

#         # # Loads depths - TODO do this properly
#         # self.depths = torch.load(os.path.join(self.root_dir, scenes[0], "depths", "depths.pt")).detach()
#         # self.correspondence_weights = torch.load(os.path.join(self.root_dir, scenes[0], "depths", "correspondence_weights.pt")).detach()
#         self.tform_depth = self.initialize_depth_tform()

#     def _get_im(self, image: UInt8[Tensor, "..."]) ->  Float[Tensor, "height width 3"]:
#         pil_im = Image.open(BytesIO(image.numpy().tobytes()))
#         # im_raw = rearrange(to_tensor(pil_im), 'c h w -> h w c')
#         return self.tform_im(pil_im)

#     def _get_flow(
#             self,
#             flow_raw: Float[Tensor, "raw_height raw_width 3"],
#             flow_mask_raw: Float[Tensor, "raw_height raw_width 3"]
#         ) -> tuple[Float[Tensor, "height width xy=2"], Float[Tensor, "height width"]]:
#         return self._preprocess_flow(flow_raw, flow_mask_raw)
    
#     @cached_property
#     def index(self) -> dict[str, Path]:
#         merged_index = {}
#         data_stages = [self.split]
#         if len(self.scenes) == 1:
#             data_stages = ("test", "train")
#         for data_stage in data_stages:
#             # Load the root's index.
#             with (self.root_dir / data_stage / "index.json").open("r") as f:
#                 index = json.load(f)
#             index = {k: Path(self.root_dir / data_stage / v) for k, v in index.items()}

#             # The constituent datasets should have unique keys.
#             assert not (set(merged_index.keys()) & set(index.keys()))

#             # Merge the root's index into the main index.
#             merged_index = {**merged_index, **index}
#         return merged_index

#     def shuffle(self, lst) -> None:
#         indices = torch.randperm(len(lst))
#         return [lst[x] for x in indices]
#         assert False
#         indices = torch.randperm(len(self.pairs))
#         self.pairs = [self.pairs[id] for id in indices]
#         self.flow_fwd_paths = [self.flow_fwd_paths[id] for id in indices]
#         self.flow_bwd_paths = [self.flow_bwd_paths[id] for id in indices]
#         self.flow_fwd_mask_paths = [self.flow_fwd_mask_paths[id] for id in indices]
#         self.flow_bwd_mask_paths = [self.flow_bwd_mask_paths[id] for id in indices]

#     def initialize_depth_tform(self) -> transforms.Compose:
#         assert any([isinstance(t, transforms.Resize) for t in self.tform_im.transforms]), "Add a torchvision.transforms.Resize transformation!"
#         assert any([isinstance(t, transforms.CenterCrop) for t in self.tform_im.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

#         for t in self.tform_im.transforms:
#             if isinstance(t, transforms.Resize):
#                 new_size = t.size
#             elif isinstance(t, transforms.CenterCrop):
#                 crop_size = t.size

#         depth_transforms = [
#             transforms.Lambda(lambda flow: rearrange(flow , 'h w -> () h w')),
#             partial(transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)),
#             transforms.Lambda(lambda flow: rearrange(flow , '() h w -> h w')),
#             transforms.CenterCrop(crop_size),
#             transforms.Lambda(lambda depth: torch.stack([depth]*3, dim=2))
#         ]


#         depth_transforms = transforms.Compose(depth_transforms)
#         return depth_transforms
    
#     def _get_depth(self, depth: Float[Tensor, "raw_height raw_width"]) -> Float[Tensor, "height width 3"]:
#         return self.tform_depth(depth)
    
#     def __iter__(self):
#         """
#         Defines dataset samples (pairs of paths of consecutive images) according to the split in a list
#         """

#         if self.split in ("train", "validation"):
#             self.scene_chunks = self.shuffle(self.scene_chunks)

#         # # When testing, the data loaders alternate chunks.
#         # worker_info = torch.utils.data.get_worker_info()
#         # if self.split == "test" and worker_info is not None:
#         #     self.scene_chunks = [
#         #         chunk
#         #         for chunk_index, chunk in enumerate(self.scene_chunks)
#         #         if chunk_index % worker_info.num_workers == worker_info.id
#         #     ]

#         # self.pairs, self.flow_fwd_paths, self.flow_bwd_paths, self.flow_fwd_mask_paths, self.flow_bwd_mask_paths = list(), list(), list(), list(), list()
#         for k in range(len(self.scene_chunks)):

#             scene_chunk = torch.load(self.scene_chunks[k])
#             # if self.cfg.overfit_to_scene is not None:
#             if len(self.scenes) == 1:
#                 item = [x for x in scene_chunk if x["key"] == self.scenes[0]]
#                 assert len(item) == 1, "Multi scene Re10kDiffmapDataset not yet implemented"
#                 # scene_chunk = item * len(scene_chunk)
#                 scene_chunk = item

#             # Shuffle chunks - TODO handle the multiscene case.
#             if self.split in ("train", "validation"):
#                 # assert len(self.scenes) == 1, "Multi scene Re10kDiffmapDataset not yet implemented"
#                 scene_chunk = self.shuffle(scene_chunk)

            
#             # Iterate 
#             for scene in scene_chunk:
#                 scene_id = scene["key"]
#                 N = len(scene["images"])

#                 # Load scene data.
#                 frames = scene["images"]
#                 fwd_flows = scene["fwd_flows"] # (B, H, W, 2)
#                 bwd_flows = scene["bwd_flows"] # (B, H, W, 2)
#                 masks_fwd_flow = scene["masks_fwd_flow"] # (B, H, W)
#                 masks_bwd_flow = scene["masks_bwd_flow"] # (B, H, W)
#                 ctxt_depths = scene["ctxt_depths"] # (B, H, W)
#                 trgt_depths = scene["trgt_depths"] # (B, H, W)
#                 correspondence_weights = scene["correspondence_weights"] # (B-1, H, W)

#                 id = torch.randint(
#                     N - 1,
#                     size=tuple(),
#                 ).item()


#                 # # Shuffle inside the chunk - TODO handle the multiscene case.
#                 # if self.split in ("train", "validation"):
#                 #     indices = torch.randperm(len(frames) - 1)
#                 # else:
#                 #     indices = torch.arange(len(frames) - 1)


#                 sample = {}
#                 sample['indices'] = torch.tensor([id, id+1])

#                 # Load image.
#                 sample[self.ctxt_key] = self._get_im(frames[id])
#                 sample[self.trgt_key] = self._get_im(frames[id+1])

#                 # # Load flow.
#                 # fwd_flow, mask_fwd_flow = self._get_flow(fwd_flows[id], masks_fwd_flow[id])
#                 # bwd_flow, mask_bwd_flow = self._get_flow(bwd_flows[id], masks_bwd_flow[id])

#                 # sample.update({
#                 #     'optical_flow': fwd_flow,
#                 #     'optical_flow_bwd': bwd_flow,
#                 #     'optical_flow_mask': mask_fwd_flow,
#                 #     'optical_flow_bwd_mask': mask_bwd_flow
#                 #     }
#                 # )

#                 # # Load depth.
#                 # sample['depth_ctxt'] = self._get_depth(ctxt_depths[id])
#                 # sample['depth_trgt'] = self._get_depth(trgt_depths[id])
#                 # sample['correspondence_weights'] = self._get_correspondence_weights(correspondence_weights[id])

#                 yield sample

#                 # # Sample pairs.
#                 # for id in indices:
#                 #     data = {}
#                 #     data['indices'] = torch.tensor([id, id+1])

#                 #     # Load image.
#                 #     data[self.ctxt_key] = self._get_im(frames[id])
#                 #     data[self.trgt_key] = self._get_im(frames[id+1])

#                 #     # Load flow.
#                 #     fwd_flow, mask_fwd_flow = self._get_flow(fwd_flows[id], masks_fwd_flow[id])
#                 #     bwd_flow, mask_bwd_flow = self._get_flow(bwd_flows[id], masks_bwd_flow[id])

#                 #     data.update({
#                 #         'optical_flow': fwd_flow,
#                 #         'optical_flow_bwd': bwd_flow,
#                 #         'optical_flow_mask': mask_fwd_flow,
#                 #         'optical_flow_bwd_mask': mask_bwd_flow
#                 #         }
#                 #     )

#                 #     # Load depth.
#                 #     data['depth_ctxt'] = self._get_depth(ctxt_depths[id])
#                 #     data['depth_trgt'] = self._get_depth(trgt_depths[id])
#                 #     data['correspondence_weights'] = self._get_correspondence_weights(correspondence_weights[id])

#                 #     yield data
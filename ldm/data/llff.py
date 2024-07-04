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
from einops import rearrange
from functools import partial

from .diffmap import DiffmapDataset

class LLFFDiffmapDataset(DiffmapDataset, Dataset):
    def __init__(self,
        root_dir: str,
        image_transforms: list = [],
        split: str = 'train',
        scenes: list | ListConfig | str | int | None = None,
        val_scenes: list | ListConfig | str | int | None = None,
        stride: int = 1,
        n_future = 1,
    ) -> None:

        assert val_scenes is not None or split != 'validation'

        DiffmapDataset.__init__(self, root_dir, image_transforms, split)
        IterableDataset.__init__(self)

        self.root_dir = Path(root_dir)
        self.split = split
        self.stride = stride
        self.n_future = n_future

        # Load scenes.
        val_scenes = self.load_scenes(val_scenes) if val_scenes is not None else []
        if self.split == 'validation':
            self.scenes = val_scenes

        elif split == 'train':
            if scenes is not None:
                self.scenes = self.load_scenes(scenes)
            else:
                self.scenes = sorted([path.name for path in self.root_dir.iterdir() if path.is_dir()])
            # Remove val scenes.
            for scene in val_scenes:
                if scene in self.scenes:
                    self.scenes.remove(scene)
            assert len(self.scenes) > 0
        
        print("Scenes : ", self.scenes)
        # Prepare pairs of frames and flows.
        self.frame_pair_paths, self.flow_fwd_paths, self.flow_bwd_paths, self.flow_fwd_mask_paths, self.flow_bwd_mask_paths = list(), list(), list(), list(), list()
        for k in range(len(self.scenes)):
            scene = self.scenes[k]

            # Load flow and image paths.
            if self.stride == 1:
                paths_frames = sorted((self.root_dir / scene / "images_diffmap_projection").iterdir())
                paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_projection", "flow_fwd_*.pt")))
                paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_projection", "flow_bwd_*.pt")))
                paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_projection", "mask_flow_fwd*.pt")))
                paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_projection", "mask_flow_bwd*.pt")))
            elif self.stride == 3:
                paths_frames = sorted((self.root_dir / scene / "images_diffmap_projection_stride_3").iterdir())
                paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_projection_stride_3", "flow_fwd_*.pt")))
                paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_projection_stride_3", "flow_bwd_*.pt")))
                paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward_projection_stride_3", "mask_flow_fwd*.pt")))
                paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward_projection_stride_3", "mask_flow_bwd*.pt")))
            else:
                raise Exception(f'Stride {self.stride} not valid, should be 1 or 3.')
            
            # Define dataset pairs.
            self.frame_pair_paths += [
                ([Path(paths_frames[idx])], [Path(p) for p in paths_frames[idx + self.stride : idx + self.n_future * self.stride + 1 : self.stride]])
                for idx in range(len(paths_frames) - (self.n_future * self.stride))
            ]
            self.flow_fwd_paths += [
                [Path(p) for p in paths_flow_fwd[idx : idx + self.n_future * self.stride : stride]]
                for idx in range(len(paths_frames) - (self.n_future * self.stride))
            ]
            self.flow_bwd_paths += [
                [Path(p) for p in paths_flow_bwd[idx : idx + self.n_future * self.stride : stride]]
                for idx in range(len(paths_frames) - (self.n_future * self.stride))
            ]
            self.flow_fwd_mask_paths += [
                [Path(p) for p in paths_flow_fwd_mask[idx : idx + self.n_future * self.stride : stride]]
                for idx in range(len(paths_frames) - (self.n_future * self.stride))
            ]
            self.flow_bwd_mask_paths += [
                [Path(p) for p in paths_flow_bwd_mask[idx : idx + self.n_future * self.stride : stride]]
                for idx in range(len(paths_frames) - (self.n_future * self.stride))
            ]

            assert len(self.frame_pair_paths) == len(self.flow_fwd_paths) == len(self.flow_bwd_paths) == len(self.flow_fwd_mask_paths) == len(self.flow_bwd_mask_paths)

        # # Loads depths - TODO do this properly
        # self.depths = torch.load(os.path.join(root_dir, scenes[0], "depths", "depths.pt")).detach() # (B,N,H,W), B=1
        # self.tform_depth = self.initialize_depth_tform()

    def load_scenes(self, raw_scenes: list | ListConfig | str | int) -> list[str]:
        if isinstance(raw_scenes, (list, ListConfig)):
            scenes = sorted(list(set([str(scene) for scene in raw_scenes])))
        elif isinstance(raw_scenes, str) or isinstance(raw_scenes, int):
            scenes = [str(raw_scenes)]
        else:
            raise AssertionError(f"Scenes field must be str or list in config, got {type(raw_scenes)}.")

        return scenes

    def initialize_depth_tform(self) -> transforms.Compose:
        assert any([isinstance(t, transforms.Resize) for t in self.tform_im.transforms]), "Add a torchvision.transforms.Resize transformation!"
        assert any([isinstance(t, transforms.CenterCrop) for t in self.tform_im.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

        for t in self.tform_im.transforms:
            if isinstance(t, transforms.Resize):
                new_size = t.size
            elif isinstance(t, transforms.CenterCrop):
                crop_size = t.size

        depth_transforms = [
            transforms.Lambda(lambda flow: rearrange(flow , 'h w -> () h w')),
            partial(transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)),
            transforms.Lambda(lambda flow: rearrange(flow , '() h w -> h w')),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda depth: torch.stack([depth]*3, dim=2))
        ]
        depth_transforms = transforms.Compose(depth_transforms)
        return depth_transforms
    
    def _get_im(self, filename: Path) -> Float[Tensor, "height width 3"]:
        im = Image.open(filename).convert("RGB")
        return self._preprocess_im(im)
    
    def _get_flow(self, flow_path: Path, flow_mask_path: Path) -> tuple[Float[Tensor, "height width 3"], Float[Tensor, "height width"]]:
        # Load flow and mask.
        flow_raw = torch.load(flow_path) #(H,W,C=2)
        mask_flow_raw = torch.load(flow_mask_path)

        # Apply transformations.
        flow, flow_mask = self._preprocess_flow(flow_raw, mask_flow_raw)
        return flow, flow_mask
    
    # def _get_depth(self, index: int) -> Float[Tensor, "height width 3"]:
    #     depth_raw = self.depths[index,:,:] # (H, W)
    #     return self.tform_depth(depth_raw)
    
    def __len__(self) -> int:
        return len(self.frame_pair_paths)

    def __getitem__(self, index: int) -> dict[Float[Tensor, "..."]]:
        # Define paths and indices.
        prev_im_path, curr_im_path = self.frame_pair_paths[index]
        fwd_flow_path = self.flow_fwd_paths[index]
        bwd_flow_path = self.flow_bwd_paths[index]
        fwd_flow_mask_path = self.flow_fwd_mask_paths[index]
        bwd_flow_mask_path = self.flow_bwd_mask_paths[index]

        # Load target, context frames.
        data = {}
        data['indices'] = torch.tensor([index, index + self.stride])
        data['ctxt_rgb'] = self._get_im(prev_im_path)
        data['trgt_rgb'] = self._get_im(curr_im_path)

        # Load flow
        flow_fwd, flow_fwd_mask = self._get_flow(fwd_flow_path, fwd_flow_mask_path)
        flow_bwd, flow_bwd_mask = self._get_flow(bwd_flow_path, bwd_flow_mask_path)

        data.update({
            'fwd_flow': flow_fwd,
            'bwd_flow': flow_bwd,
            'fwd_flow_mask': flow_fwd_mask,
            'bwd_flow_mask': flow_bwd_mask
            }
        )
        # # Load depth
        # data['ctxt_depth'] = self._load_depth(prev_idx)
        # data['trgt_depth'] = self._load_depth(curr_idx)

        return data


# class LLFFDiffmapDataset(DiffmapDataset, IterableDataset):
#     """
#     Iterable LLFF Diffmap dataset class.
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

#         # Define dataset scenes and samples.
#         if scenes is None:
#             self.scenes =  [path.name for path in self.root_dir.iterdir() if path.is_dir()]
#         else:
#             if isinstance(scenes, str):
#                 scenes = [scenes]
#             self.scenes = [str(scene) for scene in scenes]

#         self.prepare_pairs(n_val_samples_scene)
#         if self.split == "train":
#             self.shuffle()

#         # Loads depths - TODO do this properly
#         assert len(self.scenes) == 1, "Multi scene LLFFDiffmapDataset not yet implemented"
#         self.depths = torch.load(os.path.join(root_dir, scenes[0], "depths", "depths.pt")).detach()
#         self.correspondence_weights = torch.load(os.path.join(root_dir, scenes[0], "depths", "correspondence_weights.pt")).detach()
#         self.tform_depth, self.correspondence_weights_transforms = self.initialize_depth_tform()
    
#     def __len__(self) -> int:
#         return len(self.pairs)
    
#     def _get_im(self, filename: str) -> Float[Tensor, "height width 3"]:
#         raw_im = Image.open(filename).convert("RGB")
#         im = self._preprocess_im(raw_im)
#         return im
    
#     def _get_flow(
#             self,
#             flow_path: str,
#             flow_mask_path: str
#         ) -> tuple[Float[Tensor, "height width 3"], Float[Tensor, "height width"]]:
#         # Load flow and mask.
#         flow_raw = torch.load(flow_path) #(H,W,C=2)

#         # flow = flow / 0.0213

#         mask_flow_raw = torch.load(flow_mask_path)

#         flow, flow_mask = self._preprocess_flow(flow_raw, mask_flow_raw)
#         return flow, flow_mask

#     def prepare_pairs(self, n_val_samples_scene: int) -> None:

#         self.pairs_idx, self.pairs, self.flow_fwd_paths, self.flow_bwd_paths, self.flow_fwd_mask_paths, self.flow_bwd_mask_paths = list(), list(), list(), list(), list(), list()
#         for k in range(len(self.scenes)):
#             scene = self.scenes[k]

#             # Load flow and image paths.
#             paths = sorted((self.root_dir / scene / "images_diffmap").iterdir())
#             paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward", "flow_fwd_*.pt")))
#             paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward", "flow_bwd_*.pt")))
#             paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward", "mask_flow_fwd*.pt")))
#             paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward", "mask_flow_bwd*.pt")))
#             assert len(paths_flow_fwd) == len(paths_flow_bwd) and len(paths_flow_bwd) == len(paths) - 1
            
#             # Define dataset pairs.
#             all_pairs_idx = np.stack([np.arange(len(paths)-1), np.arange(1,len(paths))], axis=1)
#             val_idx = np.linspace(0,len(paths)-2, n_val_samples_scene, dtype=np.uint) #indices of val samples
#             val_pairs_idx = np.stack([val_idx, val_idx+1], axis=1) #val pairs 

#             if self.split == "validation":
#                 pairs_idx = val_pairs_idx
#             else:
#                 val_pairs_idx = np.stack([val_idx, val_idx+1], axis=1) #val pairs indices, (N,2)
#                 #indices of pairs in val_pairs_idx that contain any image from a val pair, (N,)
#                 exc_idx = np.any(
#                     np.apply_along_axis(
#                         lambda x: np.any( (all_pairs_idx==x) | (all_pairs_idx==x[::-1]), axis=1 ), #checks occurences of val images
#                         axis=1, 
#                         arr=val_pairs_idx, #slices val_pairs_idx along pairs axis
#                     ),
#                 axis=0)
#                 #excludes training pairs with val images
#                 pairs_idx = all_pairs_idx[~exc_idx]
            
#             self.pairs_idx += [(pair[0].item(), pair[1].item()) for pair in pairs_idx]
#             self.pairs += [(paths[pair[0]], paths[pair[1]]) for pair in pairs_idx]
#             self.flow_fwd_paths += [paths_flow_fwd[pair[0]] for pair in pairs_idx]
#             self.flow_bwd_paths += [paths_flow_bwd[pair[0]] for pair in pairs_idx]
#             self.flow_fwd_mask_paths += [paths_flow_fwd_mask[pair[0]] for pair in pairs_idx]
#             self.flow_bwd_mask_paths += [paths_flow_bwd_mask[pair[0]] for pair in pairs_idx]

#     def initialize_depth_tform(self) -> None:
#         assert any([isinstance(t, transforms.Resize) for t in self.tform_im.transforms]), "Add a torchvision.transforms.Resize transformation!"
#         assert any([isinstance(t, transforms.CenterCrop) for t in self.tform_im.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

#         for t in self.tform_im.transforms:
#             if isinstance(t, transforms.Resize):
#                 new_size = t.size
#             elif isinstance(t, transforms.CenterCrop):
#                 crop_size = t.size

#         depth_transforms = [
#             NormalizeDepth(),
#             ResizeDepth(new_size),
#             transforms.CenterCrop(crop_size),
#         ]
#         correspondence_weights_transforms = [
#             ResizeDepth(new_size),
#             transforms.CenterCrop(crop_size),
#             transforms.Lambda(lambda depth: torch.stack([depth]*3, dim=2))
#         ]
#         depth_transforms = transforms.Compose(depth_transforms)
#         correspondence_weights_transforms = transforms.Compose(correspondence_weights_transforms)
#         return depth_transforms, correspondence_weights_transforms
    
#     def shuffle(self) -> None:
#         indices = torch.randperm(len(self.pairs))
#         self.pairs_idx = [self.pairs_idx[x] for x in indices]
#         self.pairs = [self.pairs[x] for x in indices]
#         self.flow_fwd_paths = [self.flow_fwd_paths[x] for x in indices]
#         self.flow_bwd_paths = [self.flow_bwd_paths[x] for x in indices]
#         self.flow_fwd_mask_paths = [self.flow_fwd_mask_paths[x] for x in indices]
#         self.flow_bwd_mask_paths = [self.flow_bwd_mask_paths[x] for x in indices]

#     def _get_depth(self, index: int) -> Float[Tensor, "height width 3"]:
#         depth_raw = self.depths[index,:,:] # (H, W)
#         return self.tform_depth(depth_raw)
    
#     def _get_correspondence_weights(self, index: int) ->  Float[Tensor, "height width"]:
#         # Load and preprocess correspondence weights.
#         correspondence_weights = self.correspondence_weights[index,:,:] # (H, W)
#         return self.correspondence_weights_transforms(correspondence_weights)

#     def __iter__(self):

#         for index in range(len(self.pairs)):
#             # Define paths and indices.
#             prev_im_path = self.pairs[index][0]
#             curr_im_path = self.pairs[index][1]
#             fwd_flow_path = self.flow_fwd_paths[index]
#             bwd_flow_path = self.flow_bwd_paths[index]
#             fwd_flow_mask_path = self.flow_fwd_mask_paths[index]
#             bwd_flow_mask_path = self.flow_bwd_mask_paths[index]
#             prev_idx, curr_idx = self.pairs_idx[index][0], self.pairs_idx[index][1]
            
#             # Load target, context frames.
#             data = {}
#             data['indices'] = torch.tensor([prev_idx, curr_idx])
#             data[self.trgt_key] = self._get_im(curr_im_path)
#             data[self.ctxt_key] = self._get_im(prev_im_path)

#             # Load flow.
#             flow_fwd, flow_fwd_mask = self._get_flow(fwd_flow_path, fwd_flow_mask_path)
#             flow_bwd, flow_bwd_mask = self._get_flow(bwd_flow_path, bwd_flow_mask_path)

#             data.update({
#                 'optical_flow': flow_fwd,
#                 'optical_flow_bwd': flow_bwd,
#                 'optical_flow_mask': flow_fwd_mask,
#                 'optical_flow_bwd_mask': flow_bwd_mask
#                 }
#             )

#             # Load depth.
#             data['depth_trgt'] = self._get_depth(curr_idx)
#             data['depth_ctxt'] = self._get_depth(prev_idx)
#             data['correspondence_weights'] = self._get_correspondence_weights(prev_idx)

#             yield data





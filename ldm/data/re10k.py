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

from .diffmap import DiffmapDataset
import numpy as np
import glob
import os, os.path

from .simple import ResizeDepth, NormalizeDepth

to_tensor = tf.ToTensor()


class Re10kDiffmapDataset(DiffmapDataset, IterableDataset):
    """
    Iterable Re10k Diffmap dataset class.
    """

    def __init__(
        self,
        root_dir: str,
        image_transforms: list = [],
        trgt_key: str = "trgt",
        ctxt_key: str = "ctxt",
        split: str = "train",
        scenes: list[str] | str | None = None,
        n_val_samples_scene: int = 1
    ) -> None:
        DiffmapDataset.__init__(self, root_dir, image_transforms, trgt_key, ctxt_key, split)
        IterableDataset.__init__(self)

        # Define dataset scenes, chunk paths and samples.

        # for root in cfg.roots:
        #     root = root / self.data_stage
        #     root_chunks = sorted(
        #         [path for path in root.iterdir() if path.suffix == ".torch"]
        #     )
        #     self.scene_chunks.extend(root_chunks)

        # Collect chunks.
        assert isinstance(scenes, str) or len(scenes) == 1, "Multi scene Re10kDiffmapDataset not yet implemented"
        # TODO - modify path and don't enforce a single train sequence set for generalizable setting
        self.scene_chunks = sorted(
            [path for path in (self.root_dir / "train").iterdir() if path.suffix == ".torch"]
        )

        # if self.cfg.overfit_to_scene is not None:
        #     chunk_path = self.index[self.cfg.overfit_to_scene]
        #     self.scene_chunks = [chunk_path] * len(self.scene_chunks)

        # Define scenes and associated chunk paths
        if scenes is None:
            raise Exception('None value for scene param of dataset not yet implemented - use str or list[str].')
            # self.scenes =  [path.name for path in self.root_dir.iterdir() if path.is_dir()]
        else:
            if isinstance(scenes, str):
                self.scenes = [scenes]
                # chunk_path = self.index[scenes]
                # self.scene_chunks = [chunk_path]
            else:
                self.scenes = [str(scene) for scene in scenes]
                # self.scene_chunks = [self.index[self.cfg.overfit_to_scene]]
            try:
                self.scene_chunks = [self.index[scene] for scene in self.scenes]
            except KeyError:
                raise Exception(f'Dataset scene values should match Re10k scene keys, got unexpected value.')


        # self.prepare_pairs(n_val_samples_scene)
        # if self.split == "train":
        #     self.shuffle()

        # # Loads depths - TODO do this properly
        # self.depths = torch.load(os.path.join(self.root_dir, scenes[0], "depths", "depths.pt")).detach()
        # self.correspondence_weights = torch.load(os.path.join(self.root_dir, scenes[0], "depths", "correspondence_weights.pt")).detach()
        self.tform_depth, self.correspondence_weights_transforms = self.initialize_depth_tform()

    def _get_im(self, image: UInt8[Tensor, "..."]) ->  Float[Tensor, "height width 3"]:
        pil_im = Image.open(BytesIO(image.numpy().tobytes()))
        # im_raw = rearrange(to_tensor(pil_im), 'c h w -> h w c')
        return self.tform_im(pil_im)

    def _get_flow(
            self,
            flow_raw: Float[Tensor, "raw_height raw_width 3"],
            flow_mask_raw: Float[Tensor, "raw_height raw_width 3"]
        ) -> tuple[Float[Tensor, "height width xy=2"], Float[Tensor, "height width"]]:
        return self._preprocess_flow(flow_raw, flow_mask_raw)
    
    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.split]
        if len(self.scenes) == 1:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            # Load the root's index.
            with (self.root_dir / data_stage / "index.json").open("r") as f:
                index = json.load(f)
            index = {k: Path(self.root_dir / data_stage / v) for k, v in index.items()}

            # The constituent datasets should have unique keys.
            assert not (set(merged_index.keys()) & set(index.keys()))

            # Merge the root's index into the main index.
            merged_index = {**merged_index, **index}
        return merged_index

    def shuffle(self) -> None:
        assert False
        indices = torch.randperm(len(self.pairs))
        self.pairs = [self.pairs[id] for id in indices]
        self.flow_fwd_paths = [self.flow_fwd_paths[id] for id in indices]
        self.flow_bwd_paths = [self.flow_bwd_paths[id] for id in indices]
        self.flow_fwd_mask_paths = [self.flow_fwd_mask_paths[id] for id in indices]
        self.flow_bwd_mask_paths = [self.flow_bwd_mask_paths[id] for id in indices]

    def initialize_depth_tform(self) -> None:
        assert any([isinstance(t, transforms.Resize) for t in self.tform_im.transforms]), "Add a torchvision.transforms.Resize transformation!"
        assert any([isinstance(t, transforms.CenterCrop) for t in self.tform_im.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

        for t in self.tform_im.transforms:
            if isinstance(t, transforms.Resize):
                new_size = t.size
            elif isinstance(t, transforms.CenterCrop):
                crop_size = t.size

        depth_transforms = [
            NormalizeDepth(),
            ResizeDepth(new_size),
            transforms.CenterCrop(crop_size),
        ]
        correspondence_weights_transforms = [
            ResizeDepth(new_size),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda depth: torch.stack([depth]*3, dim=2))
        ]
        depth_transforms = transforms.Compose(depth_transforms)
        correspondence_weights_transforms = transforms.Compose(correspondence_weights_transforms)
        return depth_transforms, correspondence_weights_transforms
    
    def _get_depth(self, depth: Float[Tensor, "raw_height raw_width"]) -> Float[Tensor, "height width 3"]:
        return self.tform_depth(depth)
    
    def __iter__(self):
        """
        Defines dataset samples (pairs of paths of consecutive images) according to the split in a list
        """

        # self.pairs, self.flow_fwd_paths, self.flow_bwd_paths, self.flow_fwd_mask_paths, self.flow_bwd_mask_paths = list(), list(), list(), list(), list()
        for k in range(len(self.scene_chunks)):
            
            # scene = self.scenes[k]
            scene_chunk = torch.load(self.scene_chunks[k])
            # if self.cfg.overfit_to_scene is not None:
            if len(self.scenes) == 1:
                item = [x for x in scene_chunk if x["key"] == self.scenes[0]]
                assert len(item) == 1, "Multi scene Re10kDiffmapDataset not yet implemented"
                # scene_chunk = item * len(scene_chunk)
                scene_chunk = item

            # Shuffle chunks - TODO handle the multiscene case.
            if self.split in ("train", "validation"):
                assert len(self.scenes) == 1, "Multi scene Re10kDiffmapDataset not yet implemented"
                # scene_chunk = self.shuffle(scene_chunk)



            # # When testing, the data loaders alternate chunks.
            # worker_info = torch.utils.data.get_worker_info()
            # if self.split == "test" and worker_info is not None:
            #     self.scene_chunks = [
            #         chunk
            #         for chunk_index, chunk in enumerate(self.scene_chunks)
            #         if chunk_index % worker_info.num_workers == worker_info.id
            #     ]
            
            # Iterate 
            for scene in scene_chunk:
                scene_id = scene["key"]
                N = len(scene["images"])

                # Load scene data.
                frames = scene["images"]
                fwd_flows = scene["fwd_flows"] # (B, H, W, 2)
                bwd_flows = scene["bwd_flows"] # (B, H, W, 2)
                masks_fwd_flow = scene["masks_fwd_flow"] # (B, H, W)
                masks_bwd_flow = scene["masks_bwd_flow"] # (B, H, W)
                ctxt_depths = scene["ctxt_depths"] # (B, H, W)
                trgt_depths = scene["trgt_depths"] # (B, H, W)
                correspondence_weights = scene["correspondence_weights"] # (B-1, H, W)

                # Shuffle inside the chunk - TODO handle the multiscene case.
                if self.split in ("train", "validation"):
                    indices = torch.randperm(len(frames) - 1)
                else:
                    indices = torch.arange(len(frames) - 1)

                # Sample pairs.
                for id in indices:
                    data = {}
                    data['indices'] = torch.tensor([id, id+1])

                    # Load image.
                    data[self.ctxt_key] = self._get_im(frames[id])
                    data[self.trgt_key] = self._get_im(frames[id+1])

                    # Load flow.
                    fwd_flow, mask_fwd_flow = self._get_flow(fwd_flows[id], masks_fwd_flow[id])
                    bwd_flow, mask_bwd_flow = self._get_flow(bwd_flows[id], masks_bwd_flow[id])

                    data.update({
                        'optical_flow': fwd_flow,
                        'optical_flow_bwd': bwd_flow,
                        'optical_flow_mask': mask_fwd_flow,
                        'optical_flow_bwd_mask': mask_bwd_flow
                        }
                    )

                    # Load depth.
                    data['depth_ctxt'] = self._get_depth(ctxt_depths[id])
                    data['depth_trgt'] = self._get_depth(trgt_depths[id])
                    data['correspondence_weights'] = self._get_correspondence_weights(correspondence_weights[id])

                    yield data
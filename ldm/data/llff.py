from torch.utils.data import Dataset
from .simple import make_tranforms, ResizeFlow, ResizeDepth, NormalizeDepth
from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf.listconfig import ListConfig
import os, os.path
import torch
from einops import rearrange
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import glob


class LLFFnfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        image_key='trgt',
        cond_key='ctxt',
        split='train',
        scenes=None,
        n_val_samples_scene=1
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.image_key = image_key
        self.cond_key = cond_key
        self.split = split

        #dataset scenes
        if scenes is None:
            self.scenes =  [path.name for path in self.root_dir.iterdir() if path.is_dir()]
        else:
            if isinstance(scenes, str):
                scenes = [scenes]
            self.scenes = [str(scene) for scene in scenes]

        self.prepare_pairs(n_val_samples_scene)
        self.tform = make_tranforms(image_transforms)

    def prepare_pairs(self, n_val_samples_scene):
        '''
        Defines dataset samples (pairs of paths of consecutive images) according to the split in a list
        '''

        self.pairs = list()
        for k in range(len(self.scenes)):
            scene = self.scenes[k]

            paths = sorted((self.root_dir / scene / "images").iterdir())

            #defines dataset pairs
            all_pairs_idx = np.stack([np.arange(len(paths)-1), np.arange(1,len(paths))], axis=1)
            val_idx = np.linspace(0,len(paths)-2, n_val_samples_scene, dtype=np.uint) #indices of val samples
            val_pairs_idx = np.stack([val_idx, val_idx+1], axis=1) #val pairs 

            if self.split == 'val':
                pairs_idx = val_pairs_idx
            else:
                val_pairs_idx = np.stack([val_idx, val_idx+1], axis=1) #val pairs indices, (N,2)
                #indices of pairs in val_pairs_idx that contain any image from a val pair, (N,)
                exc_idx = np.any(
                    np.apply_along_axis(
                        lambda x: np.any( (all_pairs_idx==x) | (all_pairs_idx==x[::-1]), axis=1 ), #checks occurences of val images
                        axis=1, 
                        arr=val_pairs_idx, #slices val_pairs_idx along pairs axis
                    ),
                axis=0)
                #excludes training pairs with val images
                pairs_idx = all_pairs_idx[~exc_idx]

            self.pairs += [(paths[j[0]], paths[j[1]]) for j in pairs_idx]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        prev = self.pairs[index][0]
        curr = self.pairs[index][1]
        data = {}
        data[self.image_key] = self._load_im(curr)
        data[self.cond_key] = self._load_im(prev)
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)
    

class LLFFDiffmapDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        image_key='trgt',
        cond_key='ctxt',
        split='train',
        scenes=None,
        n_val_samples_scene=1,
        ) -> None:

        self.root_dir = Path(root_dir)
        self.image_key = image_key
        self.cond_key = cond_key
        self.split = split

        #dataset scenes
        if scenes is None:
            self.scenes =  [path.name for path in self.root_dir.iterdir() if path.is_dir()]
        else:
            if isinstance(scenes, str):
                scenes = [scenes]
            self.scenes = [str(scene) for scene in scenes]

        self.prepare_pairs(n_val_samples_scene)

        #defines transforms
        self.tform = make_tranforms(image_transforms)
        self.tform_flow, self.tform_flow_mask = self.initialize_flow_tform()
        self.tform_depth, self.correspondence_weights_transforms = self.initialize_depth_tform()

        #loads flow and depth - TODO do this properly
        assert len(self.scenes) == 1, "Multi scene LLFFDiffmapDataset not yet implemented"
        self.depths = torch.load(os.path.join(root_dir, scenes[0], "depths", "depths.pt")).detach() # (B,N,H,W), B=1
        self.correspondence_weights = torch.load(os.path.join(root_dir, scenes[0], "depths", "correspondence_weights.pt")).detach() # (B,N-1,H,W), B=1


    def initialize_flow_tform(self):
        assert any([isinstance(t, transforms.Resize) for t in self.tform.transforms]), "Add a torchvision.transforms.Resize transformation!"
        assert any([isinstance(t, transforms.CenterCrop) for t in self.tform.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

        for t in self.tform.transforms:
            if isinstance(t, transforms.Resize):
                new_size = t.size
            elif isinstance(t, transforms.CenterCrop):
                crop_size = t.size

        flow_transforms = [
            transforms.Lambda(lambda x: rearrange(x , 'h w c -> c h w')),
            ResizeFlow(new_size),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda x: rearrange(x , 'c h w -> h w c')),            
        ]
        flow_mask_transforms = [
            ResizeFlow(new_size),
            transforms.CenterCrop(crop_size),
        ]
        flow_transforms = transforms.Compose(flow_transforms)
        flow_mask_transforms = transforms.Compose(flow_mask_transforms)
        return flow_transforms, flow_mask_transforms
    
    def initialize_depth_tform(self):
        assert any([isinstance(t, transforms.Resize) for t in self.tform.transforms]), "Add a torchvision.transforms.Resize transformation!"
        assert any([isinstance(t, transforms.CenterCrop) for t in self.tform.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

        for t in self.tform.transforms:
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
        ]
        depth_transforms = transforms.Compose(depth_transforms)
        correspondence_weights_transforms = transforms.Compose(correspondence_weights_transforms)
        return depth_transforms, correspondence_weights_transforms

    def prepare_pairs(self, n_val_samples_scene):
        '''
        Defines dataset samples (pairs of paths of consecutive images) according to the split in a list
        '''

        self.pairs = list()
        for k in range(len(self.scenes)):
            scene = self.scenes[k]

            # Load flow and image paths.
            paths = sorted((self.root_dir / scene / "images_diffmap").iterdir())
            paths_flow_fwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward", "flow_fwd_*.pt")))
            paths_flow_bwd = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward", "flow_bwd_*.pt")))
            paths_flow_fwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_forward", "mask_flow_fwd*.pt")))
            paths_flow_bwd_mask = sorted(glob.glob(os.path.join(self.root_dir, scene, "flow_backward", "mask_flow_bwd*.pt")))
            assert len(paths_flow_fwd) == len(paths_flow_bwd) and len(paths_flow_bwd) == len(paths) - 1
            
            # Define dataset pairs.
            all_pairs_idx = np.stack([np.arange(len(paths)-1), np.arange(1,len(paths))], axis=1)
            val_idx = np.linspace(0,len(paths)-2, n_val_samples_scene, dtype=np.uint) #indices of val samples
            val_pairs_idx = np.stack([val_idx, val_idx+1], axis=1) #val pairs 

            if self.split == 'val':
                self.pairs_idx = val_pairs_idx
            else:
                val_pairs_idx = np.stack([val_idx, val_idx+1], axis=1) #val pairs indices, (N,2)
                #indices of pairs in val_pairs_idx that contain any image from a val pair, (N,)
                exc_idx = np.any(
                    np.apply_along_axis(
                        lambda x: np.any( (all_pairs_idx==x) | (all_pairs_idx==x[::-1]), axis=1 ), #checks occurences of val images
                        axis=1, 
                        arr=val_pairs_idx, #slices val_pairs_idx along pairs axis
                    ),
                axis=0)
                #excludes training pairs with val images
                self.pairs_idx = all_pairs_idx[~exc_idx]

            self.pairs += [(paths[j[0]], paths[j[1]]) for j in self.pairs_idx]
            self.flow_fwd_paths = [paths_flow_fwd[j[0]] for j in self.pairs_idx]
            self.flow_bwd_paths = [paths_flow_bwd[j[0]] for j in self.pairs_idx]
            self.flow_fwd_mask_paths = [paths_flow_fwd_mask[j[0]] for j in self.pairs_idx]
            self.flow_bwd_mask_paths = [paths_flow_bwd_mask[j[0]] for j in self.pairs_idx]


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        #paths and indices
        prev_im_path = self.pairs[index][0]
        curr_im_path = self.pairs[index][1]
        fwd_flow_path = self.flow_fwd_paths[index]
        bwd_flow_path = self.flow_bwd_paths[index]
        fwd_flow_mask_path = self.flow_fwd_mask_paths[index]
        bwd_flow_mask_path = self.flow_bwd_mask_paths[index]
        prev_idx, curr_idx = self.pairs_idx[index,0].item(), self.pairs_idx[index,1].item()
        
        # Load target, context frames
        data = {}
        data['indices'] = torch.tensor([prev_idx, curr_idx])
        data[self.image_key] = self._load_im(curr_im_path)
        data[self.cond_key] = self._load_im(prev_im_path)

        # Load flow
        flow_fwd, flow_fwd_mask = self._load_flow(fwd_flow_path, fwd_flow_mask_path)
        flow_bwd, flow_bwd_mask = self._load_flow(bwd_flow_path, bwd_flow_mask_path)

        data.update({
            'optical_flow': flow_fwd,
            'optical_flow_bwd': flow_bwd,
            'optical_flow_mask': flow_fwd_mask,
            'optical_flow_bwd_mask': flow_bwd_mask
            }
        )
        # Load depth
        data['depth_trgt'] = self._load_depth(curr_idx)
        data['depth_ctxt'] = self._load_depth(prev_idx)
        data['correspondence_weights'] = self._load_correspondence_weights(prev_idx)

        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)
  
    def _load_depth(self, index):
        depth = self.depths[0,index,:,:]
        depth_transformed = self.tform_depth(depth) #(H,W)
        #extends to three channels to be processed like an image
        depth_transformed = torch.stack([depth_transformed]*3, dim=2) #(H,W,C)
        return depth_transformed
    
    def _load_correspondence_weights(self, index):
        correspondence_weights = self.correspondence_weights[0,index,:,:]
        depth_weights_transformed = self.correspondence_weights_transforms(correspondence_weights) #(H,W)
        return depth_weights_transformed


    def _load_flow(self, flow_path, flow_mask_path):
        # Load flow and mask.
        flow = torch.load(flow_path) #(H,W,C=2)
        flow_mask = torch.load(flow_mask_path)

        # Apply transformations.
        flow_transformed = self.tform_flow(flow) #(H,W,C=2)
        flow_mask_transformed = self.tform_flow_mask(flow_mask[None]).squeeze(0) #(H,W)

        # Add a third channel.
        flow_transformed = torch.cat([flow_transformed, torch.zeros_like(flow_transformed[:,:,0,None])], dim=2) #(H,W,C=3)
        return flow_transformed, flow_mask_transformed





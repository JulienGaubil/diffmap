import torch
import numpy as np

from jaxtyping import Float
from torch import Tensor
from pathlib import Path
from torchvision import transforms
from einops import rearrange

from .llff import LLFFDiffmapDataset
from .utils.io import load_exr
from .utils.camera import Intrinsics, Extrinsics, Camera
from .utils.tforms import ResizeIntrinsics, CenterCropIntrinsics

class RoomsDiffmapDataset(LLFFDiffmapDataset):
    def __init__(
            self,
            flip_trajectories: bool = False,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.flip_trajectories = flip_trajectories

        self.depth_pairs_paths = list()
        self.cameras_pairs = list()
        for k in range(len(self.scenes)):
            scene = self.scenes[k]

            depth_paths = sorted((self.root_dir / scene / "depth_exr").iterdir())
            self.depth_pairs_paths += [
                (Path(depth_paths[idx]), [Path(p) for p in depth_paths[idx + self.stride : idx + self.n_future * self.stride + 1 : self.stride]])
                for idx in range(len(depth_paths) - (self.n_future + self.stride) + 1)
            ]

            # Prepare intrinsics.
            intrinsics_file = self.root_dir / scene / "intrinsics.txt"
            intrinsics_pinhole = np.loadtxt(intrinsics_file, max_rows=1)
            fx = fy = intrinsics_pinhole[0]
            cx, cy = intrinsics_pinhole[1], intrinsics_pinhole[2]
            HW = np.loadtxt(intrinsics_file, skiprows=3)#.astype(int)
            intrinsics = Intrinsics(**{'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'resolution': HW})

            # Prepare extrinsics.
            extrinsics_file = self.root_dir / scene / "poses.txt"
            c2ws = np.loadtxt(extrinsics_file)
            c2ws = torch.from_numpy(c2ws.reshape((-1,4,4)))
            w2cs = torch.linalg.inv(c2ws)
            extrinsics = [Extrinsics(R=w2cs[i,:3,:3], t=w2cs[i,:3,3]) for i in range(w2cs.size(0))]

            # Load camera pairs.
            cameras = [Camera(intrinsics=intrinsics, extrinsics=extrinsics[i]) for i in range(len(extrinsics))]
            self.cameras_pairs += [
                (cameras[idx], [cam for cam in cameras[idx + self.stride  : idx + self.n_future * self.stride + 1 : self.stride]])
                for idx in range(len(cameras) - (self.n_future + self.stride) + 1)
            ]

        assert len(self.depth_pairs_paths) == len(self.frame_pair_paths) == len(self.cameras_pairs)
        
        # Create transforms.
        self.tform_depth, self.tform_correspondence_weights = self.initialize_depth_tform()
        self.tform_intrinsics = self.initialize_intrinsics_tform()

    def initialize_intrinsics_tform(self) -> transforms.Compose:
        assert any([isinstance(t, transforms.Resize) for t in self.tform_im.transforms]), "Add a torchvision.transforms.Resize transformation!"
        assert any([isinstance(t, transforms.CenterCrop) for t in self.tform_im.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

        for t in self.tform_im.transforms:
            if isinstance(t, transforms.Resize):
                new_size = t.size
            elif isinstance(t, transforms.CenterCrop):
                crop_size = t.size

        # Initialize intrinsics transformations.
        intrinsics_transforms = [
            ResizeIntrinsics(new_size),
            CenterCropIntrinsics(crop_size),
        ]
        intrinsics_transforms = transforms.Compose(intrinsics_transforms)
        return intrinsics_transforms

    def _get_flow(self, flow_paths: list[Path], flow_mask_paths: list[Path]) -> tuple[Float[Tensor, "n_future height width 3"], Float[Tensor, "n_future height width"]]:
        flows = list()
        flow_masks = list()
        for k in range(len(flow_paths)):
            # Load flow and mask.
            flow_raw = torch.load(flow_paths[k]) #(H,W,C=2)
            mask_flow_raw = torch.load(flow_mask_paths[k])

            # Apply transformations.
            flow, flow_mask = self._preprocess_flow(flow_raw, mask_flow_raw)
            flows.append(flow)
            flow_masks.append(flow_mask)
        
        flows = torch.stack(flows, dim=0)
        flow_masks = torch.stack(flow_masks, dim=0)
        return flows, flow_masks

    def _get_depth(self, path: Path) -> Float[Tensor, "height width 3"]:
        depth_raw = load_exr(path) # (H, W)
        return self.tform_depth(depth_raw)
    
    def _get_camera_pair(self, index: int) -> tuple[Camera, Camera]:
        prev_camera, future_cameras = self.cameras_pairs[index]

        # Apply intrinsics transformations.
        prev_camera.intrinsics = self.tform_intrinsics(prev_camera.intrinsics)
        for k in range(self.n_future):
            future_cameras[k].intrinsics = self.tform_intrinsics(future_cameras[k].intrinsics)

        return prev_camera, future_cameras
        
    # def _get_correspondence_weights(self, index: int) ->  Float[Tensor, "height width"]:
    #     # Load and preprocess correspondence weights.
    #     correspondence_weights = self.correspondence_weights[index,:,:] # (H, W)
    #     return self.tform_correspondence_weights(correspondence_weights)

    def __getitem__(self, index: int) -> dict[Float[Tensor, "..."]]:

        if self.flip_trajectories:
            pair_idx = np.random.permutation(2)
        else:
            pair_idx = np.arange(2)

        # Define paths and indices.
        if pair_idx[0] == 0:
            prev_im_path, future_im_paths = self.frame_pair_paths[index]
            prev_depth_path, future_depth_paths = self.depth_pairs_paths[index]
            fwd_flow_paths = self.flow_fwd_paths[index]
            bwd_flow_paths = self.flow_bwd_paths[index]
            fwd_flow_mask_paths = self.flow_fwd_mask_paths[index]
            bwd_flow_mask_paths = self.flow_bwd_mask_paths[index]
            prev_camera, future_cameras = self._get_camera_pair(index)
            indices = torch.arange(index, index + (self.stride * self.n_future) + 1, self.stride)
        else:
            # Flipping trajectory
            fwd_flow_paths = self.flow_bwd_paths[index][::-1]
            bwd_flow_paths = self.flow_fwd_paths[index][::-1]
            fwd_flow_mask_paths = self.flow_bwd_mask_paths[index][::-1]
            bwd_flow_mask_paths = self.flow_fwd_mask_paths[index][::-1]

            # Invert order for future frames.
            prev_camera, next_cameras = self._get_camera_pair(index)
            prev_im_path, next_im_paths = self.frame_pair_paths[index]
            prev_depth_path, next_depth_paths = self.depth_pairs_paths[index]

            future_cameras = next_cameras[:-1][::-1] + [prev_camera]
            future_im_paths = next_im_paths[:-1][::-1] + [prev_im_path]
            future_depth_paths = next_depth_paths[:-1][::-1] + [prev_depth_path]

            prev_camera = next_cameras[-1]
            prev_im_path = next_im_paths[-1]
            prev_depth_path = next_depth_paths[-1]
            indices = torch.arange(index, index + (self.stride * self.n_future) + 1, self.stride).flip(dims=[0])

        data = {}

        # Load target, context frames.
        ctxt_ims = rearrange(self._get_im(prev_im_path), 'h w c -> () h w c')
        trgt_ims = torch.stack([self._get_im(p) for p in future_im_paths], dim=0)
        data.update({
            self.ctxt_key: ctxt_ims,
            self.trgt_key: trgt_ims,
            'indices': indices
            }
        )

        # Load flow
        flows_fwd, flow_fwd_masks = self._get_flow(fwd_flow_paths, fwd_flow_mask_paths)
        flows_bwd, flow_bwd_masks = self._get_flow(bwd_flow_paths, bwd_flow_mask_paths)
        data.update({
            'optical_flow': flows_fwd,
            'optical_flow_bwd': flows_bwd,
            'optical_flow_mask': flow_fwd_masks,
            'optical_flow_bwd_mask': flow_bwd_masks
            }
        )

        # Load depths and correspondence weights.
        depths_ctxt = rearrange(self._get_depth(prev_depth_path), 'h w c -> () h w c')
        depths_trgt = torch.stack([self._get_depth(p) for p in future_depth_paths], dim=0)
        weights = torch.ones_like(depths_trgt[:,:,:,0]) # TODO - remove hack
        data.update({
            'depth_ctxt': depths_ctxt,
            'depth_trgt': depths_trgt,
            'correspondence_weights': weights
            }
        )

        # Load cameras.
        data.update({
            'camera_ctxt': prev_camera,
            'camera_trgt': future_cameras
            }
        )
        
        return data









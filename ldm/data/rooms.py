import torch
import numpy as np

from jaxtyping import Float
from torch import Tensor
from pathlib import Path
from torchvision import transforms

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

            # Prepare depths paths.
            depth_paths = sorted((self.root_dir / scene / "depth_exr").iterdir())[::self.stride]
            self.depth_pairs_paths += [(Path(depth_paths[idx]), Path(depth_paths[idx+1])) for idx in range(len(depth_paths) - 1)]

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
            c2ws = torch.from_numpy(c2ws.reshape((-1,4,4)))[::self.stride]
            assert c2ws.shape[0] == len(depth_paths)
            w2cs = torch.linalg.inv(c2ws)
            extrinsics = [Extrinsics(R=w2cs[i,:3,:3], t=w2cs[i,:3,3]) for i in range(w2cs.size(0))]

            # Load camera pairs.
            cameras = [Camera(intrinsics=intrinsics, extrinsics=extrinsics[i]) for i in range(len(extrinsics))]
            self.cameras_pairs += [(cameras[idx], cameras[idx+1]) for idx in range(len(cameras) - 1)]
        
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

    def _get_depth(self, path: Path) -> Float[Tensor, "height width 3"]:
        depth_raw = load_exr(path) # (H, W)
        return self.tform_depth(depth_raw)
    
    def _get_camera_pair(self, index: int) -> tuple[Camera, Camera]:
        prev_camera, curr_camera = self.cameras_pairs[index]

        # Apply intrinsics transformations.
        prev_camera.intrinsics = self.tform_intrinsics(prev_camera.intrinsics)
        curr_camera.intrinsics = self.tform_intrinsics(curr_camera.intrinsics)

        return prev_camera, curr_camera
        
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
        prev_im_path, curr_im_path = [self.frame_pair_paths[index][k] for k in pair_idx]
        prev_depth_path, curr_depth_path = [self.depth_pairs_paths[index][k] for k in pair_idx]
        if pair_idx[0] == 0:
            fwd_flow_path = self.flow_fwd_paths[index]
            bwd_flow_path = self.flow_bwd_paths[index]
            fwd_flow_mask_path = self.flow_fwd_mask_paths[index]
            bwd_flow_mask_path = self.flow_bwd_mask_paths[index]
            prev_camera, curr_camera = self._get_camera_pair(index)
        else:
            # Flipping trajectory
            fwd_flow_path = self.flow_bwd_paths[index]
            bwd_flow_path = self.flow_fwd_paths[index]
            fwd_flow_mask_path = self.flow_bwd_mask_paths[index]
            bwd_flow_mask_path = self.flow_fwd_mask_paths[index]
            curr_camera, prev_camera = self._get_camera_pair(index)


        # Load target, context frames.
        data = {}
        data['indices'] = torch.tensor([index, index + 1])[pair_idx]
        data[self.ctxt_key] = self._get_im(prev_im_path)
        data[self.trgt_key] = self._get_im(curr_im_path)

        # print(data['indices'])
        # print(prev_im_path, curr_im_path)
        # print(fwd_flow_path, bwd_flow_path)
        # print(fwd_flow_mask_path, bwd_flow_mask_path)
        # print(prev_depth_path, curr_depth_path)
        # print('')

        # Load flow
        flow_fwd, flow_fwd_mask = self._get_flow(fwd_flow_path, fwd_flow_mask_path)
        flow_bwd, flow_bwd_mask = self._get_flow(bwd_flow_path, bwd_flow_mask_path)
        data.update({
            'optical_flow': flow_fwd,
            'optical_flow_bwd': flow_bwd,
            'optical_flow_mask': flow_fwd_mask,
            'optical_flow_bwd_mask': flow_bwd_mask
            }
        )

        # Load depths and correspondence weights.
        data.update({
            'depth_ctxt': self._get_depth(prev_depth_path),
            'depth_trgt': self._get_depth(curr_depth_path)
        })
        # TODO - remove hack
        data['correspondence_weights'] = torch.ones_like(data['depth_ctxt'][:,:,0])

        # Load cameras.
        data.update({
            'camera_ctxt': prev_camera,
            'camera_trgt': curr_camera
        })
        
        return data









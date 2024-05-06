import torch
from jaxtyping import Float, Int
from torch import Tensor
from pathlib import Path

from .llff import LLFFDiffmapDataset
from .utils.io import load_exr

class RoomsDiffmapDataset(LLFFDiffmapDataset):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.depth_pairs_paths = list()
        for k in range(len(self.scenes)):
            scene = self.scenes[k]
            depth_paths = sorted((self.root_dir / scene / "depth_exr").iterdir())
            self.depth_pairs_paths += [(Path(depth_paths[idx]), Path(depth_paths[idx+1])) for idx in range(len(depth_paths) - 1)]
        
        self.tform_depth, self.correspondence_weights_transforms = self.initialize_depth_tform()


    def _get_depth(self, path: Path) -> Float[Tensor, "height width 3"]:
        depth_raw = load_exr(path) # (H, W)
        return self.tform_depth(depth_raw)
    
    # def _get_correspondence_weights(self, index: int) ->  Float[Tensor, "height width"]:
    #     # Load and preprocess correspondence weights.
    #     correspondence_weights = self.correspondence_weights[index,:,:] # (H, W)
    #     return self.correspondence_weights_transforms(correspondence_weights)
    
    def __getitem__(self, index: int) -> dict[Float[Tensor, "..."]]:
        data = super().__getitem__(index)

        # Load depths and correspondence weights.
        prev_depth_path = self.depth_pairs_paths[index][0]
        curr_depth_path = self.depth_pairs_paths[index][1]

        data.update({
            'depth_ctxt': self._get_depth(prev_depth_path),
            'depth_trgt': self._get_depth(curr_depth_path)
        })
        # TODO - remove hack
        data['correspondence_weights'] = torch.ones_like(data['depth_ctxt'][:,:,0])
        
        return data









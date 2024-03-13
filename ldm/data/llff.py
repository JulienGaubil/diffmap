from torch.utils.data import Dataset
from .simple import make_tranforms
from pathlib import Path
from PIL import Image
import numpy as np
from omegaconf.listconfig import ListConfig



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
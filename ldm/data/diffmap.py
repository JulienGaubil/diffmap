
import torch

from pathlib import Path
from abc import abstractmethod
from abc import ABC, abstractmethod
from typing import Any
from jaxtyping import Float
from torch import Tensor
from einops import rearrange
from torchvision import transforms
from omegaconf import ListConfig

from ldm.util import instantiate_from_config
from .simple import ResizeFlow


class DiffmapDataset(ABC):
    """
    Abstract Diffmap dataset class.
    """
    root_dir: Path
    trgt_key: str
    ctxt_key: str
    split: str

    tform_im: transforms.transforms.Compose
    tform_flow: transforms.transforms.Compose
    tform_flow_mask: transforms.transforms.Compose

    def __init__(
        self,
        root_dir: str,
        image_transforms: list,
        trgt_key: str, # TODO - do it properly, target signal key in batch
        ctxt_key: str, # TODO - do it properly, conditional signal key in batch
        split: str,
    ) -> None:

        # Common attributes.
        self.root_dir = Path(root_dir)
        self.trgt_key = trgt_key
        self.ctxt_key = ctxt_key
        self.split = split

        # Define transforms.
        self.tform_im = self.initialize_im_tform(image_transforms)
        self.tform_flow, self.tform_flow_mask = self.initialize_flow_tform()
    
    @abstractmethod
    def _get_im(self, *args) -> Float[Tensor, "height width 3"]:
        """
        Loads and preprocess image.
        """
        pass

    @abstractmethod
    def _get_flow(self, *args) -> tuple[Float[Tensor, "height width 3"], Float[Tensor, "height width"]]:
        """
        Loads raw optical flow and confidence flow mask.
        """
        pass

    def initialize_im_tform(self, image_transforms) -> None:
        # Instantiate image transforms defined in config.
        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]

        image_transforms.append(transforms.ToTensor())
        # Diffusion image transforms.
        image_transforms.append(
            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))
        )
        # Compose image transforms.
        image_transforms = transforms.Compose(image_transforms)
        return image_transforms
    
    def initialize_flow_tform(self) -> None:
        assert any([isinstance(t, transforms.Resize) for t in self.tform_im.transforms]), "Add a torchvision.transforms.Resize transformation!"
        assert any([isinstance(t, transforms.CenterCrop) for t in self.tform_im.transforms]), "Add a torchvision.transforms.CenterCrop transformation!"

        for t in self.tform_im.transforms:
            if isinstance(t, transforms.Resize):
                new_size = t.size
            elif isinstance(t, transforms.CenterCrop):
                crop_size = t.size

        flow_transforms = [
            transforms.Lambda(lambda flow: rearrange(flow , 'h w c -> c h w')),
            ResizeFlow(new_size),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda flow: rearrange(flow , 'c h w -> h w c')),
            transforms.Lambda(lambda flow: torch.cat([flow, torch.zeros_like(flow[:,:,0,None])], dim=2))
        ]
        flow_mask_transforms = [
            transforms.Lambda(lambda mask_flow: mask_flow.unsqueeze(0)),
            ResizeFlow(new_size),
            transforms.CenterCrop(crop_size),
            transforms.Lambda(lambda mask_flow: mask_flow.squeeze(0))            
        ]
        flow_transforms = transforms.Compose(flow_transforms)
        flow_mask_transforms = transforms.Compose(flow_mask_transforms)
        return flow_transforms, flow_mask_transforms
    
    def _preprocess_im(
            self,
            im_raw: Float[Tensor, "height width 3"],
        ) -> Float[Tensor, "transformed_height transformed_width 3"]:
        # Apply image transformations.
        return self.tform_im(im_raw)
    
    def _preprocess_flow(
            self,
            flow_raw: Float[Tensor, "raw_height raw_width xy=2"],
            flow_mask_raw:  Float[Tensor, "raw_height raw_width"]
        ) -> tuple[Float[Tensor, "height width 3"], Float[Tensor, "height width"]]:
        # Apply flow transformations.
        flow = self.tform_flow(flow_raw)
        flow_mask = self.tform_flow_mask(flow_mask_raw)
        return flow, flow_mask
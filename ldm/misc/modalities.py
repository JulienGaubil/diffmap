import torch
import numpy as np

from typing import Literal
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from omegaconf import ListConfig

from ldm.misc.util import instantiate_from_config, default


@dataclass
class Modality:
    name: str
    modality: Literal["rgb", "flow", "depth"]
    multiplicity: int
    channels_m: int # number of input/output channels per modality without multiplicity
    denoised: bool


class Modalities:

    def __init__(self, modalities_cfg_list: ListConfig):
        
        # Instanciate modalities list.
        # assert isinstance(modalities_cfg_list, list), f'type modalities cfg : {type(modalities_cfg_list)}'
        self.modality_list = [instantiate_from_config(modality_cfg) for modality_cfg in modalities_cfg_list]
        self.modality_list.sort(key=lambda modality: modality.name)

        assert sorted(list(set(self.names))) == self.names, 'Duplicate found in modalities names'

    @property
    def denoised_modalities(self) -> list[Modality]:
        return [modality for modality in self.modality_list if modality.denoised]
    
    @property
    def clean_modalities(self) -> list[Modality]:
        return [modality for modality in self.modality_list if not modality.denoised]
    
    @property
    def names(self) -> list[str]:
        return [modality.name for modality in self.modality_list]
    
    @property
    def names_denoised(self) -> list[str]:
        return [modality.name for modality in self.modality_list if modality.denoised]

    @property
    def names_clean(self) -> list[str]:
        return [modality.name for modality in self.modality_list if not modality.denoised]

    @property
    def n_channels(self) -> int:
        return sum([modality.channels_m * modality.multiplicity for modality in self.modality_list])
    
    @property
    def n_noisy_channels(self) -> int:
        return sum([modality.channels_m * modality.multiplicity for modality in self.modality_list if modality.denoised])

    @property
    def n_clean_channels(self) -> int:
        return sum([modality.channels_m * modality.multiplicity for modality in self.modality_list if not modality.denoised])
    
    @property
    def modality_idxs(self) -> list:
        channels_list = [modality.channels_m * modality.multiplicity for modality in self.modality_list]
        return np.cumsum(np.asarray(channels_list)).tolist()
    
    def split_modalities(
        self,
        x: Float[Tensor, "... channels height width"],
        modality_names: list[str] | str | None = None,
        dim: int = 2
    ) -> dict[str, Float[Tensor, "... (multiplicity channels_m) height width"]]:
        '''Split input tensor given modalities.
        '''
        if isinstance(modality_names, str):
            modality_names = [modality_names]
        modality_names = sorted(default(modality_names, self.names))
        
        assert all([name_m in self.names for name_m in modality_names]), f'Unknown modality name in {modality_names}, should be {self.names}.'
        modality_list = [self[name_m] for name_m in modality_names]
        
        n_channels = sum([modality.channels_m * modality.multiplicity for modality in modality_list])
        assert x.size(dim) == n_channels, f"Input size {x.size()} doesn't match expected number of channels {n_channels} along dim {dim}."
        C = [modality.channels_m *  modality.multiplicity for modality in modality_list]
        split_dict = dict(zip(modality_names, torch.split(x, C, dim=dim)))

        return split_dict
    
    def split_modalities_multiplicity(
        self,
        x: Float[Tensor, "... multiplicity channel height width"],
        modality_names: list[str] | str | None = None,
        dim: int = 1
    ) -> dict[str, Float[Tensor, "... (multiplicity channels_m) height width"]]:
        '''Split input tensor along multiplicity dim given modalities.
        '''
        if isinstance(modality_names, str):
            modality_names = [modality_names]
        modality_names = sorted(default(modality_names, self.names))
        
        assert all([name_m in self.names for name_m in modality_names]), f'Unknown modality name in {modality_names}, should be {self.names}.'
        modality_list = [self[name_m] for name_m in modality_names]
        
        n_frames = sum([modality.multiplicity for modality in modality_list])
        assert x.size(dim) == n_frames, f"Input size {x.size()} doesn't match expected number of frames {n_frames} along dim {dim}."
        C = [modality.multiplicity for modality in modality_list]
        split_dict = dict(zip(modality_names, torch.split(x, C, dim=dim)))

        return split_dict
    
    def split_noisy_clean(
        self,
        x: Float[Tensor, "... channels height width"],
        dim: int = 2
    ) -> tuple[Float[Tensor, "... noisy_channels height width"], Float[Tensor, "... clean_channels height width"]]:
        split_all = self.split_modalities(x, dim=dim)
        
        dummy_shape = list(x.shape)
        dummy_shape[dim] = 0
        if self.n_noisy_channels > 0:
            noisy = torch.cat(
                [split_all[name_m] for name_m in self.names_denoised],
                dim=dim
            )
        else:
            noisy = torch.zeros(dummy_shape)

        if self.n_clean_channels > 0:
            clean = torch.cat(
                [split_all[name_m] for name_m in self.names_clean],
                dim=dim
            )
        else:
            clean = torch.zeros(dummy_shape)
        return noisy, clean

    def cat_modalities(
        self,
        x_dict: dict,
        dim: int = 2,
        modality_names: list[str] | str | None = None
    ) -> Float[Tensor, "... channels height width"]:
        if isinstance(modality_names, str):
            modality_names = [modality_names]
        modality_names = sorted(default(modality_names, list(x_dict.keys())))

        assert all([name_m in self.names for name_m in modality_names]), f'Unknown modality name in {modality_names}, should be {self.names}.'
        assert all([name_m in x_dict.keys() for name_m in modality_names]), f'Input modalities {modality_names}, not in dict, should be {x_dict.keys()}.'

        ccat = torch.cat(
            [x_dict[name_m] for name_m in modality_names],
            dim=dim
        )

        return ccat

    def __getitem__(self, name_m: str) -> Modality:
        assert name_m in self.names
        idx = self.names.index(name_m)
        return self.modality_list[idx]
    
    def __len__(self) -> int:
        return len(self.modality_list)
    
    # def get_modality_idx(
    #     self,
    #     name_m: str
    # ) -> tuple[int, int]:
    #     '''Get start and end indices for a modality in tensor.
    #     '''
    #     assert name_m in self.names
    #     idx = self.names.index(name_m)
    #     modalities_idxs = [0] + self.modality_idxs
    #     start_idx, end_idx = modalities_idxs[idx:idx+1]

    #     return start_idx, end_idx

    # def split_noisy_clean(
    #     self,

    # )


    

        


    




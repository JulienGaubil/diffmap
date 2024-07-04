import torch
import numpy as np

from typing import Literal
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from omegaconf import ListConfig, DictConfig

from ldm.misc.util import instantiate_from_config, default


@dataclass
class Modality:
    name: str
    modality: Literal["rgb", "flow", "depth"]
    multiplicity: int
    channels_m: int # number of input/output channels per modality without multiplicity
    denoised: bool

    @property
    def _id(self):
        return '_'.join([self.name, self.modality]) if self.modality != "" else self.name

    def __hash__(self):
        return hash(self._id)
    
    def __eq__(self, other):
        if not isinstance(other, Modality):
            return NotImplemented
        return self._id == other._id


class Modalities:
    def __init__(self, modalities_cfg_list: ListConfig | list = []):
        
        # Instanciate modalities list.
        if isinstance(modalities_cfg_list, list):
            self._modality_list = modalities_cfg_list
        else:
            modality_list = [instantiate_from_config(modality_cfg) for modality_cfg in modalities_cfg_list]
            if all([isinstance(mod, Modality) for mod in modality_list]):
                self._modality_list = modality_list
            elif all([isinstance(mod, Modalities) for mod in modality_list]):
                self._modality_list = []
                for k in range(len(modality_list)):
                    self._modality_list += modality_list[k].modality_list
            else:
                raise Exception('Modalities input list should all Modality or all Modalities objects.')

        assert sorted(list(set(self.ids))) == self.ids, 'Duplicate found in modalities ids'
    
    @property
    def modality_list(self):
        return sorted(list(set(self._modality_list)), key=lambda modality: modality._id)

    @property
    def denoised_modalities(self) -> list[Modality]:
        return [modality for modality in self.modality_list if modality.denoised]
    
    @property
    def clean_modalities(self) -> list[Modality]:
        return [modality for modality in self.modality_list if not modality.denoised]
    
    @property
    def ids(self) -> list[str]:
        return [modality._id for modality in self.modality_list]
    
    @property
    def ids_denoised(self) -> list[str]:
        return [modality._id for modality in self.modality_list if modality.denoised]

    @property
    def ids_clean(self) -> list[str]:
        return [modality._id for modality in self.modality_list if not modality.denoised]

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
    
    def __getattr__(self, name):
        return getattr(self._modality_list, name)
    
    def __add__(self, other):
        if isinstance(other, Modalities):
            return Modalities(self._modality_list + other._modality_list)
        return NotImplemented

    def __iter__(self):
        return iter(self.modality_list)
    
    def __next__(self):
        return next(self.modality_list)

    def split_modalities(
        self,
        x: Float[Tensor, "... channels height width"],
        modality_ids: list[str] | str | None = None,
        dim: int = 1
    ) -> dict[str, Float[Tensor, "... (multiplicity channels_m) height width"]]:
        '''Split input tensor given modalities.
        '''
        if isinstance(modality_ids, str):
            modality_ids = [modality_ids]
        modality_ids = sorted(default(modality_ids, self.ids))
        
        assert all([id_m in self.ids for id_m in modality_ids]), f'Unknown modality id in {modality_ids}, should be {self.ids}.'
        modality_list = [self[id_m] for id_m in modality_ids]
        
        n_channels = sum([modality.channels_m * modality.multiplicity for modality in modality_list])
        assert x.size(dim) == n_channels, f"Input size {x.size()} doesn't match expected number of channels {n_channels} along dim {dim}."
        C = [modality.channels_m *  modality.multiplicity for modality in modality_list]
        split_dict = dict(zip(modality_ids, torch.split(x, C, dim=dim)))

        return split_dict
    
    def split_modalities_multiplicity(
        self,
        x: Float[Tensor, "... _ channel height width"],
        modality_ids: list[str] | str | None = None,
        dim: int = 1
    ) -> dict[str, Float[Tensor, "... multiplicity channels_m height width"]]:
        '''Split input tensor along multiplicity dim given modalities.
        '''
        if isinstance(modality_ids, str):
            modality_ids = [modality_ids]
        modality_ids = sorted(default(modality_ids, self.ids))
        
        assert all([id_m in self.ids for id_m in modality_ids]), f'Unknown modality id in {modality_ids}, should be {self.ids}.'
        modality_list = [self[id_m] for id_m in modality_ids]
        
        n_frames = sum([modality.multiplicity for modality in modality_list])
        assert x.size(dim) == n_frames, f"Input size {x.size()} doesn't match expected number of frames {n_frames} along dim {dim}."
        C = [modality.multiplicity for modality in modality_list]
        split_dict = dict(zip(modality_ids, torch.split(x, C, dim=dim)))

        return split_dict
    
    def split_noisy_clean(
        self,
        x: Float[Tensor, "... channels height width"],
        dim: int = 1
    ) -> tuple[Float[Tensor, "... noisy_channels height width"], Float[Tensor, "... clean_channels height width"]]:
        split_all = self.split_modalities(x, dim=dim)
        
        dummy_shape = list(x.shape)
        dummy_shape[dim] = 0
        if self.n_noisy_channels > 0:
            noisy = torch.cat(
                [split_all[id_m] for id_m in self.ids_denoised],
                dim=dim
            )
        else:
            noisy = torch.zeros(dummy_shape)

        if self.n_clean_channels > 0:
            clean = torch.cat(
                [split_all[id_m] for id_m in self.ids_clean],
                dim=dim
            )
        else:
            clean = torch.zeros(dummy_shape)
        return noisy, clean

    def cat_modalities(
        self,
        x_dict: dict,
        dim: int = 2,
        modality_ids: list[str] | str | None = None
    ) -> Float[Tensor, "... channels height width"]:
        if isinstance(modality_ids, str):
            modality_ids = [modality_ids]
        modality_ids = sorted(default(modality_ids, list(x_dict.keys())))

        assert all([id_m in self.ids for id_m in modality_ids]), f'Unknown modality id in {modality_ids}, should be {self.ids}.'
        assert all([id_m in x_dict.keys() for id_m in modality_ids]), f'Input modalities {modality_ids}, not in dict, should be {x_dict.keys()}.'

        ccat = torch.cat(
            [x_dict[id_m] for id_m in modality_ids],
            dim=dim
        )

        return ccat

    def __getitem__(self, id_m: str) -> Modality:
        assert id_m in self.ids
        idx = self.ids.index(id_m)
        return self.modality_list[idx]
    
    def __len__(self) -> int:
        return len(self.modality_list)


class FlowModalities(Modalities):
    def __init__(
        self,
        forward_cfg: DictConfig | None = None,
        backward_cfg: DictConfig | None = None,
        forward_mask_cfg: DictConfig | None = None,
        backward_mask_cfg: DictConfig | None = None,
    ) -> None:
        modalities = list()
        if forward_cfg is not None:
            modalities.append(
                Modality(
                    name='fwd',
                    modality='flow',
                    multiplicity=forward_cfg.get('multiplicity', 1),
                    channels_m=forward_cfg.get('channels_m', 2),
                    denoised=forward_cfg.get('denoised', False)
                )
            )
        
        if backward_cfg is not None:
            modalities.append(
                Modality(
                    name='bwd',
                    modality='flow',
                    multiplicity=backward_cfg.get('multiplicity', 1),
                    channels_m=backward_cfg.get('channels_m', 2),
                    denoised=backward_cfg.get('denoised', False)
                )
            )

        if forward_mask_cfg is not None:
            modalities.append(
                Modality(
                    name='mask_fwd',
                    modality='flow',
                    multiplicity=forward_mask_cfg.get('multiplicity', 1),
                    channels_m=forward_mask_cfg.get('channels_m', 1),
                    denoised=forward_mask_cfg.get('denoised', False)
                )
            )

        if backward_mask_cfg is not None:
            modalities.append(
                Modality(
                    name='mask_bwd',
                    modality='flow',
                    multiplicity=backward_mask_cfg.get('multiplicity', 1),
                    channels_m=backward_mask_cfg.get('channels_m', 1),
                    denoised=backward_mask_cfg.get('denoised', False)
                )
            )

        super().__init__(modalities)


class RGBModalities(Modalities):
    def __init__(
        self,
        ctxt_cfg: DictConfig | None = None,
        trgt_cfg: DictConfig | None = None,
    ) -> None:
        modalities = list()
        if ctxt_cfg is not None:
            modalities.append(
                Modality(
                    name='ctxt',
                    modality='rgb',
                    multiplicity=ctxt_cfg.get('multiplicity', 1),
                    channels_m=ctxt_cfg.get('channels_m', 3),
                    denoised=ctxt_cfg.get('denoised', False)
                )
            )
        
        if trgt_cfg is not None:
            modalities.append(
                Modality(
                    name='trgt',
                    modality='rgb',
                    multiplicity=trgt_cfg.get('multiplicity', 1),
                    channels_m=trgt_cfg.get('channels_m', 3),
                    denoised=trgt_cfg.get('denoised', False)
                )
            )

        super().__init__(modalities)
    

class DepthModalities(Modalities):
    def __init__(
        self,
        ctxt_cfg: DictConfig | None = None,
        trgt_cfg: DictConfig | None = None,
    ) -> None:
        modalities = list()
        if ctxt_cfg is not None:
            modalities.append(
                Modality(
                    name='ctxt',
                    modality='depth',
                    multiplicity=ctxt_cfg.get('multiplicity', 1),
                    channels_m=ctxt_cfg.get('channels_m', 1),
                    denoised=ctxt_cfg.get('denoised', False)
                )
            )
        
        if trgt_cfg is not None:
            modalities.append(
                Modality(
                    name='trgt',
                    modality='depth',
                    multiplicity=trgt_cfg.get('multiplicity', 1),
                    channels_m=trgt_cfg.get('channels_m', 1),
                    denoised=trgt_cfg.get('denoised', False)
                )
            )

        super().__init__(modalities)
    




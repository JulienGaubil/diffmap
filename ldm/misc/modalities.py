import torch
import numpy as np
import torch.nn as nn

from typing import Literal, List, Dict
from jaxtyping import Float
from torch import Tensor
from dataclasses import dataclass
from omegaconf import ListConfig, DictConfig

from ldm.misc.util import instantiate_from_config, get_obj_from_str, default


@dataclass
class Modality:
    name: str
    modality: Literal["rgb", "flow", "depth", "pointmap", "weight"]
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
    """Class that defines an ordered list of Modalities.
    Can either be defined as a list of Modality objects **or** act as a superset of subsets of Modalities objects.
    """
    def __init__(self, modalities_init: ListConfig | List[Modality] | List['Modalities'] = []) -> None:

        if isinstance(modalities_init, list):
            self._instantiate_from_list(modalities_init)
        elif isinstance(modalities_init, ListConfig):
            init_list = [instantiate_from_config(modality_cfg) for modality_cfg in modalities_init]
            self._instantiate_from_list(init_list)
        else:
            raise ValueError(f'Non-recognized instantiation param for Modalities, should be list of Modality | Modalities or ListConfig, got {type(modalities_init)}.')

    @property
    def modality_list(self) -> List[Modality]:
        return sorted(list(set(self._modality_list)), key=lambda modality: modality._id)

    @property
    def denoised_modalities(self) -> List[Modality]:
        return [modality for modality in self.modality_list if modality.denoised]
    
    @property
    def clean_modalities(self) -> List[Modality]:
        return [modality for modality in self.modality_list if not modality.denoised]
    
    @property
    def ids(self) -> List[str]:
        return [modality._id for modality in self.modality_list]
    
    @property
    def ids_denoised(self) -> List[str]:
        return [modality._id for modality in self.modality_list if modality.denoised]

    @property
    def ids_clean(self) -> List[str]:
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
    
    def __getattr__(self, name):
        if name != '_modality_list':
            try:
                return getattr(self._modality_list, name)
            except AttributeError:
                raise AttributeError(f"AttributeError: 'Modalities' object has no attribute '{name}")
        else: # when _modality_list has not yet been instantiated
            return []
    
    def __add__(self, other):
        if isinstance(other, Modalities):
            assert (self.subsets == [] and other.subsets == []) or (self.subsets != [] and other.subsets != []), f'Tried to concatenate a list of Modalities subsets with a simple list of Modalities'
            if self.subsets == []:
                return Modalities(self._modality_list + other._modality_list)
            else:
                return Modalities(self.subsets + other.subsets)

    def __iter__(self):
        return iter(self.modality_list)
    
    def __next__(self):
        return next(self.modality_list)

    def _instantiate_from_list(self, init_list: List[Modality] | List['Modalities']) -> None:
        self.subsets = list()

        # Instantiate from a list of Modality objects.
        if all(isinstance(mod, Modality) for mod in init_list):
            assert len(set(init_list)) == len(init_list), 'Found duplicate in input modalities list.'
            self._modality_list = init_list
        
        # Instantiate as a superset of a list of Modalities objects.
        elif all(isinstance(mod, Modalities) for mod in init_list):
            self._modality_list = []
            for subset in init_list:
                self.subsets.append(subset)
                assert all(modality not in self.modality_list for modality in subset), f' Found duplicate Modality in input modalities subsets.'
                self._modality_list += subset.modality_list
        else:
            raise ValueError('Config list should define either only Modality objects or only Modalities objects.')

    def split_modalities(
        self,
        x: Float[Tensor, "... (multiplicity channel) height width"],
        modality_ids: List[str] | str | None = None,
        dim: int = 1
    ) -> Dict[str, Float[Tensor, "... (multiplicity_m channels_m) height width"]]:
        """Split input tensor along gathered (multiplicty channel) dim given modalities.
        """
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
        x: Float[Tensor, "... multiplicity channel height width"],
        modality_ids: List[str] | str | None = None,
        dim: int = 1
    ) -> Dict[str, Float[Tensor, "... multiplicity_m channel height width"]]:
        """Split input tensor along multiplicity dim given modalities.
        """
        # Format input modalities.
        if isinstance(modality_ids, str):
            modality_ids = [modality_ids]
        modality_ids = sorted(default(modality_ids, self.ids))
        modality_list = [self[id_m] for id_m in modality_ids]
        
        # Prepare chunks along multiplicity dim.
        n_frames = sum([modality.multiplicity for modality in modality_list])
        C = [modality.multiplicity for modality in modality_list]
        assert x.size(dim) == n_frames, f"Input size {x.size()} doesn't match expected number of frames {n_frames} along dim {dim}."

        # Split tensor and map to dict.
        split_dict = dict(zip(modality_ids, torch.split(x, C, dim=dim)))
        return split_dict
    
    def split_noisy_clean(
        self,
        x: Float[Tensor, "... (multiplicity channel) height width"],
        dim: int = 1
    ) -> tuple[
        Float[Tensor, "... (multiplicity_noisy channel_noisy) height width"],
        Float[Tensor, "... (multiplicity_clean channel_clean) height width"]
    ]:
        """Split input tensor along multiplicity dim given modalities.
        """
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
        x_dict: Dict[str, Float[Tensor, "... (multiplicity_m channels_m) height width"]],
        dim: int = 2,
        modality_ids: List[str] | str | None = None
    ) -> Float[Tensor, "... (multiplicity channels) height width"]:
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
        assert id_m in self.ids, f'Unknown modality id {id_m}, should be one of {self.ids}.'
        idx = self.ids.index(id_m)
        return self.modality_list[idx]
    
    def __len__(self) -> int:
        return len(self.modality_list)


class SequenceModalities(Modalities):
    def __init__(
        self,
        past_modality: Modality | None = None,
        future_modality: Modality | None = None,
    ) -> None:
        """Modalities that follow a sequential order, split in past elements and future elements.
        """
        assert not (past_modality is None and future_modality is None)

        modalities = list()
        self._past_modality = past_modality
        self._future_modality = future_modality

        # Instantiate container.
        if past_modality is not None:
            modalities.append(self._past_modality)
        if future_modality is not None:
            modalities.append(self._future_modality)
        super().__init__(modalities)

    def cat_sequence_multiplicity(
        self,
        x_dict: Dict[str, Float[Tensor, "... frame_m channel_m height width"]],
        dim: int = 1,
    ) -> Float[Tensor, "... frame channel height width"]:
        """Concatenate a sequence of along multiplicity/frame dim.
        """
        geometry_sequence = list()
        for modality in [self._past_modality, self._future_modality]:
            if modality is not None:
                assert modality._id in x_dict.keys(), f'Modality {modality._id} not in dict.'
                geometry_sequence.append(x_dict[modality._id])

        return torch.cat(geometry_sequence, dim=dim)
    
    def split_sequence_multiplicity(
        self,
        x_seq: Float[Tensor, "... frame channel height width"],
        dim: int = 1,
    ) ->  Dict[str, Float[Tensor, "... frame_m channel_m height width"]]:
        """Split a sequence of along multiplicity/frame dim.
        """
        if self._past_modality is not None and self._future_modality is not None:
            n_past = self._past_modality.multiplicity
            n_future = self._future_modality.multiplicity
            assert x_seq.size(dim) == n_past + n_future, f"Sequence length {x_seq.size()} doesn't match expected sequence length {n_past} + {n_future} along dim {dim}."

            chunks = [n_past, n_future]
            modality_ids = [self._past_modality._id,  self._future_modality._id]
            return dict(zip(modality_ids, torch.split(x_seq, chunks, dim=dim)))
        
        elif self._past_modality is None:
            return {self._future_modality._id: x_seq}
        elif self._future_modality is None:
            return {self._past_modality._id: x_seq}
        
    def flip_sequence_multiplicity(
        self,
        x_dict: Dict[str, Float[Tensor, "... frame_m channel_m height width"]], # TODO handle typing well
        dim: int = 1
    ) -> Dict[str, Float[Tensor, "... frame_m channel_m height width"]]: # TODO handle typing well
        """Flip a sequence of along multiplicity/frame dim.
        """
        geometry_sequence = self.cat_sequence_multiplicity(x_dict, dim)
        flipped_sequence = torch.flip(geometry_sequence, dim=[dim])
        flipped_modalities = self.split_sequence_multiplicity(flipped_sequence, dim=dim)
        return flipped_modalities


class FlowModalities(Modalities):
    def __init__(
        self,
        forward_cfg: DictConfig | None = None,
        backward_cfg: DictConfig | None = None,
        forward_mask_cfg: DictConfig | None = None,
        backward_mask_cfg: DictConfig | None = None,
    ) -> None:
        modalities = list()

        self._fwd_flow_modality, self._bwd_flow_modality = None, None
        self._mask_fwd_flow_modality, self._mask_bwd_flow_modality = None, None

        if forward_cfg is not None:
            self._fwd_flow_modality = Modality(
                name='fwd',
                modality='flow',
                multiplicity=forward_cfg.get('multiplicity', 1),
                channels_m=forward_cfg.get('channels_m', 2),
                denoised=forward_cfg.get('denoised', False)
            )
            modalities.append(self._fwd_flow_modality)
        
        if backward_cfg is not None:
            self._bwd_flow_modality = Modality(
                name='bwd',
                modality='flow',
                multiplicity=backward_cfg.get('multiplicity', 1),
                channels_m=backward_cfg.get('channels_m', 2),
                denoised=backward_cfg.get('denoised', False)
            )
            modalities.append(self._bwd_flow_modality)

        if forward_mask_cfg is not None:
            self._mask_fwd_flow_modality = Modality(
                name='mask_fwd',
                modality='flow',
                multiplicity=forward_mask_cfg.get('multiplicity', 1),
                channels_m=forward_mask_cfg.get('channels_m', 1),
                denoised=forward_mask_cfg.get('denoised', False)
            )
            modalities.append(self._mask_fwd_flow_modality)

        if backward_mask_cfg is not None:
            self._mask_bwd_flow_modality = Modality(
                name='mask_bwd',
                modality='flow',
                multiplicity=backward_mask_cfg.get('multiplicity', 1),
                channels_m=backward_mask_cfg.get('channels_m', 1),
                denoised=backward_mask_cfg.get('denoised', False)
            )
            modalities.append(self._mask_bwd_flow_modality)

        super().__init__(modalities)


class RGBModalities(SequenceModalities):
    def __init__(
        self,
        ctxt_cfg: DictConfig | None = None,
        trgt_cfg: DictConfig | None = None,
    ) -> None:
        
        # Instantiate sequence.
        assert not (ctxt_cfg is None and trgt_cfg is None)
        past_modality, future_modality = None, None
        if ctxt_cfg is not None:
            past_modality = Modality(
                name='ctxt',
                modality='rgb',
                multiplicity=ctxt_cfg.get('multiplicity', 1),
                channels_m=ctxt_cfg.get('channels_m', 3),
                denoised=ctxt_cfg.get('denoised', False)
            )
        if trgt_cfg is not None:
            future_modality = Modality(
                name='trgt',
                modality='rgb',
                multiplicity=trgt_cfg.get('multiplicity', 1),
                channels_m=trgt_cfg.get('channels_m', 3),
                denoised=trgt_cfg.get('denoised', False)
            )
        super().__init__(past_modality, future_modality)


class GeometryModalities(SequenceModalities):
    def __init__(
        self,
        parameterization: Literal["depth", "local_pointmap", "shared_pointmap"], # local_pointmap := pointmaps local camera space, shared_pointmap := pointmaps in first camera space
        ctxt_cfg: DictConfig | None = None,
        trgt_cfg: DictConfig | None = None,
        pointmap_mapping_func: DictConfig | None = None
    ) -> None:
        # Handle 3D parameterization.
        assert parameterization in ["depth", "local_pointmap", "shared_pointmap"], f'Bad parameterization {parameterization}.'
        self.parameterization = parameterization
        self.modality_type = "pointmap" if self.parameterization in ["local_pointmap", "shared_pointmap"] else "depth"
        
        self.pointmap_mapping_func = get_obj_from_str(pointmap_mapping_func) if pointmap_mapping_func is not None else nn.Identity()

        # Instantiate sequence.
        assert not (ctxt_cfg is None and trgt_cfg is None)
        past_modality, future_modality = None, None
        if ctxt_cfg is not None:
            past_modality = Modality(
                name='ctxt',
                modality=self.modality_type,
                multiplicity=ctxt_cfg.get('multiplicity', 1),
                channels_m=ctxt_cfg.get('channels_m', 1),
                denoised=ctxt_cfg.get('denoised', False)
            )
        if trgt_cfg is not None:
            future_modality =  Modality(
                name='trgt',
                modality=self.modality_type,
                multiplicity=trgt_cfg.get('multiplicity', 1),
                channels_m=trgt_cfg.get('channels_m', 1),
                denoised=trgt_cfg.get('denoised', False)
            )
        super().__init__(past_modality, future_modality)

    def to_geometry(
        self,
        x: Float[Tensor, "sample frame 3 height width"],
        **kwargs
    ) -> Float[Tensor, "sample frame height width"] | Float[Tensor, "sample frame 3 height width"]:
        """Mapping from diffusion output to predicted geometric modality values.
        """
        if self.modality_type == "depth":
            return self.to_depth(x, **kwargs)
        elif self.modality_type == "pointmap":
            return self.pointmap_mapping_func(x, **kwargs)

    def to_depth(
        self,
        x: Float[Tensor, "sample frame 3 height width"],
        **kwargs
    ) -> Float[Tensor, "sample frame height width"]:
        """Flowmap exponential mapping for depths.
        """
        depths = x.mean(2)
        return (depths / 1000).exp() + 0.01
import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import hydra
import pytorch_lightning as pl
import json
import copy

from packaging import version
from typing import Any
from jaxtyping import Float
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import save_image
from tqdm import tqdm
from flow_vis_torch import flow_to_color
from pathlib import Path
from itertools import product
from functools import reduce
from operator import getitem

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.misc.util import instantiate_from_config, modify_conv_weights
from ldm.modules.flowmap.visualization.depth import color_map_depth
from ldm.modules.flowmap.visualization.color import apply_color_map_to_image
from ldm.data.utils.camera import pixel_grid_coordinates, to_euclidean_space, to_projective_space, Camera, K_to_intrinsics, Extrinsics
from ldm.visualization import filter_depth
from ldm.modules.flowmap.flow.flow_predictor import Flows
from ldm.misc.projection import compute_flow_projection, compute_consistency_mask

MULTINODE_HACKS = False

@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def print_tensor_dict(dictionnary: dict, root: bool = True) -> dict:
    if root:
        tmp_dict = copy.deepcopy(dictionnary)
    else:
        tmp_dict = dictionnary

    for k, v in tmp_dict.items():
        # Recursively parse dict.
        if isinstance(v, dict):
            print_tensor_dict(v, root=False)
        # Modify values to be suitable for print.
        elif isinstance(v, (Tensor,np.ndarray)):
            tmp_dict[k] = v.shape
        elif isinstance(v,list):
            if all([isinstance(s, (Tensor, np.ndarray)) for s in v]):
                tmp_dict[k] = [s.shape for s in v]
            elif all([isinstance(s, Camera) for s in v]):
                tmp_dict[k] = len(v)

    if root:
        print(json.dumps(tmp_dict, indent=4))




def prepare_visualization(value: Any, keys: list[str]) -> Float[Tensor, "batch height width 3"]:

    if 'depths' in keys:
        value, _ = filter_depth(value)
        images = color_map_depth(value)
        images = rearrange(images, 'n c h w -> n h w c')
    elif 'flows' in keys:
        if 'forward_mask' in keys or 'backward_mask' in keys:
            images = repeat(value.float(), 'n h w -> n h w c', c=3)
        else:
            images = [flow_to_color(rearrange(value[k], 'h w xy -> xy h w')) / 255 for k in range(value.shape[0])]
            images = torch.stack(images, dim=0)
            images = rearrange(images, 'n c h w -> n h w c')
    elif 'correspondence_weights' in keys:
        images = apply_color_map_to_image(value, "gray")
        images = rearrange(images, 'n c h w -> n h w c')
    else:
        images = value
    
    assert images.max() <= 1, f'{images.max()}'
    return images
    

def process_value(value: Any, keys: list[str]) -> Any:
    '''Preprocess logged values according to the modality.
    '''
    if value is not None:
        if isinstance(value, Tensor):
            value = value.cpu()
        if 'rgbs' in keys:
            # (b h w c)
            value = torch.clamp(value, -1., 1.)
            value = (value + 1.0) / 2.0
            if value.size(1) == 3:
                value = rearrange(value, 'b c h w -> b h w c')
        elif 'depths' in keys:
            # (b h w)
            if len(value.shape) == 4:
                if value.size(-1) == 3:
                    value = value.mean(-1)
                elif value.size(1) == 3:
                    value = value.mean(1)
                else:
                    value = value.squeeze()
                    if len(value.shape) == 2:
                        value = value.unsqueeze(0)
            print(0.1, keys, value.shape)
        
        elif 'flows' in keys:
            if 'forward' in keys or 'backward' in keys:
                # (b h w xy=2)
                if value.size(-1) == 3:
                    value = value[...,:2]
                elif value.size(1) == 3:
                    value = value[:,:2]
                    value = rearrange(value, 'b xy h w -> b h w xy')
                value = value * 0.0213
        return value
    return None


def get_keys(modality: str) -> tuple[list[str],list[str]]:

    if modality in ['rgbs', 'depths']:
        log_dict_keys_list = list(product(['gt', 'sampled'], ['ctxt', 'trgt'], [modality])) # e.g. ('gt', 'trgt', modality)

        if modality == 'rgbs':
            # TODO - do properly without assuming a log structure.
            batch_logs_keys_list = [('conditioning', 'concat'), ('trgt', 'inputs')] #keys for gt
            batch_logs_keys_list.extend([(), ('trgt', 'samples')]) #keys for samples
        elif modality == 'depths':
            batch_logs_keys_list = list(product(['inputs', 'samples'],['depth_ctxt', 'depth_trgt']))
            batch_logs_keys_list = [(k2,k1) for k1,k2 in batch_logs_keys_list] # e.g. ('depth_ctxt', 'samples')

    elif modality == 'flows':
        log_dict_keys_list = list(product(['gt', 'sampled'], [modality], ['forward', 'backward', 'forward_mask', 'backward_mask'])) # e.g. ('sampled', 'flows', 'forward_mask')
        batch_logs_keys_list = [('optical_flow', 'inputs'), (), (), ()] #keys for gt
        batch_logs_keys_list.extend([('optical_flow', 'samples'), (), (), ()]) #keys for samples        

    else:
        raise Exception('Modality not recognized, should be "flows", "rgbs" or "depths".')


    assert len(log_dict_keys_list) == len(batch_logs_keys_list)
    log_dict_keys_list = [list(keys) for keys in log_dict_keys_list]
    batch_logs_keys_list = [list(keys) for keys in batch_logs_keys_list]

    return log_dict_keys_list, batch_logs_keys_list


def get_value(dictionnary, keys: list[str] | str) -> Any:
    """Get value from a nested dict given a list of keys.
    """
    if isinstance(keys, str):
        keys = [keys]
    
    if len(keys) > 0:
        tmp_dict = dictionnary
        for key in keys:
            try:
                tmp_dict = tmp_dict[key]
            except KeyError:
                return None
        return tmp_dict
    else:
        return None
    

def default_set(
    dictionnary: dict,
    keys: list[str] | str,
    value: Any
) -> None:
    # Replace value in dict.
    if get_value(dictionnary,keys) is not None:
        if len(keys) > 1:
            tmp_dict = get_value(dictionnary, keys[:-1])
            tmp_dict[keys[-1]] = value
        else:
            dictionnary[keys[-1]] = value
        return None
    # Put value in dict.
    elif value is not None and len(keys) > 0:
        tmp_dict = dictionnary

        for i, key in enumerate(keys):
            try:
                tmp_dict = tmp_dict[key]
            except KeyError:
                # Create dict if not final key.
                if i < len(keys) -1:
                    tmp_dict[key] = dict()
                    tmp_dict = tmp_dict[key]
                # Initiate leaf list and exits.
                else:
                    tmp_dict[key] = value

def default_add(
    dictionnary: dict,
    keys: list[str] | str,
    value: Any,
    operation: str = "append" #defines operation to perform, either set, append or extend
) -> None:
    """Append a value in a nested dict whose final values are lists given keys.
    """
    if isinstance(keys, str):
        keys = [keys]
    
    if value is not None and len(keys) > 0:
        tmp_dict = dictionnary

        for i, key in enumerate(keys):
            try:
                tmp_dict = tmp_dict[key]
            except KeyError:
                # Create dict if not final key.
                if i < len(keys) -1:
                    tmp_dict[key] = dict()
                    tmp_dict = tmp_dict[key]
                # Initiate leaf list and exits.
                else:
                    if operation == "append":
                        tmp_dict[key] = [value]
                    elif operation == "extend":
                        tmp_dict[key] = value
                    return None

        # Extend leaf list.
        if operation == "append":
            tmp_dict.append(value)
        elif operation == "extend":
            assert isinstance(value, list)
            tmp_dict.extend(value)
        else:
            raise Exception('Operation not recognized, should be "append" or "extend".')
        return None

def default_log(logs_dict: dict, modality: str, batch_logs: dict) -> None:

    log_dict_keys_list, batch_logs_keys_list = get_keys(modality)

    for keys_logs_dict, keys_batch_logs in zip(log_dict_keys_list, batch_logs_keys_list):
        value = get_value(batch_logs, keys_batch_logs)
        if 'depths' in keys_logs_dict:
            print(0.02, value.shape)
        value = process_value(value, keys_logs_dict)
        default_add(dictionnary=logs_dict, keys=keys_logs_dict, value=value)    


@hydra.main(
        version_base=None,
        config_path='configs',
        config_name='common'
)
def sample(config: DictConfig) -> None:
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.
   
    # data:
    #   target: ldm.data.datamodule.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value

    # Force load config references.
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    # Get checkpoint infos.
    # assert config.experiment_cfg.resume is not None, "Provide a log path that contains checkpoint to sample"
    if config.experiment_cfg.resume is not None:
        if not os.path.exists(config.experiment_cfg.resume):
            raise ValueError("Cannot find {}".format(config.experiment_cfg.resume))
        if os.path.isfile(config.experiment_cfg.resume):
            paths = config.experiment_cfg.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = Path("/".join(paths[:-2]))
            ckpt = Path(config.experiment_cfg.resume)
        else:
            assert os.path.isdir(config.experiment_cfg.resume), config.experiment_cfg.resume
            logdir = Path(config.experiment_cfg.resume)
            ckpt = logdir / "checkpoints" / "last.ckpt"

        # Indicate checkpoint path for model.
        config.model.params.ckpt_path = str(ckpt)

        print(0.2, OmegaConf.to_yaml(config))

        # else:
        #     config.experiment_cfg.resume = ""
        #     if config.experiment_cfg.name:
        #         name = "_" + config.experiment_cfg.name
        #     else:
        #         name = ""
        #     nowname = now + name + config.experiment_cfg.postfix
        #     logdir = os.path.join(config.experiment_cfg.logdir, nowname)

        # TODO - load from checkpoint and config indicated by logdir?
        ckpt_cfg_path = str(logdir / "configs")
        ckpt_cfg_files = glob.glob(os.path.join(ckpt_cfg_path, "*project.yaml"))
        # print(0.3, ckpt_cfg_files)
        assert len(ckpt_cfg_files) == 1
        cfg_file = str(ckpt_cfg_files[0])
        # Former config.
        ckpt_cfg = OmegaConf.load(cfg_file)
        # print(0.4, OmegaConf.to_yaml(ckpt_cfg))

    # Load model.
    seed_everything(config.experiment_cfg.seed)

    # model
    model = instantiate_from_config(config.model)
    model.cpu()

    if not config.experiment_cfg.finetune_from == "":
        rank_zero_print(f"Attempting to load state from {config.experiment_cfg.finetune_from}")
        old_state = torch.load(config.experiment_cfg.finetune_from, map_location="cpu")
        if "state_dict" in old_state:
            rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
            old_state = old_state["state_dict"]

        #Check if we need to port input weights from 4ch input to n*4ch
        in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
        new_state = model.state_dict()
        in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
        in_shape = in_filters_current.shape
        #initializes new input weights if input dims don't match
        if in_shape != in_filters_load.shape:
            rank_zero_print("Modifying weights to multiply their number of input channels")
            keys_to_change_inputs = [
                "model.diffusion_model.input_blocks.0.0.weight",
                "model_ema.diffusion_modelinput_blocks00weight",
            ]
            
            for k in keys_to_change_inputs:
                input_weight_new = new_state[k] # size (C_out, C_in, kernel_size, kernel_size)
                C_in_new = input_weight_new.size(1)
                C_in_old = old_state[k].size(1)

                assert  C_in_new % C_in_old == 0 and  C_in_new >= C_in_old, f"Number of input channels for checkpoint and new U-Net should be multiple, got {C_in_new} and {C_in_old}"
                print("modifying input weights for compatibility")
                copy_weights = True
                scale = 1/(C_in_new//C_in_old) if copy_weights else 1e-8 #scales input to prevent activations to blow up when copying
                #repeats checkpoint weights C_in_new//C_in_old times along input channels=dim1
                old_state[k] = modify_conv_weights(old_state[k], scale=scale, n=C_in_new//C_in_old, dim=1, copy_weights=copy_weights)
                
        #check if we need to port output weights from 4ch to n*4 ch
        keys_to_change_outputs_new = [
                "model.diff_out.0.weight",
                "model.diff_out.0.bias",
                "model.diff_out.2.weight",
                "model.diff_out.2.bias",
                "model_ema.diff_out0weight",
                "model_ema.diff_out0bias",
                "model_ema.diff_out2weight",
                "model_ema.diff_out2bias",
        ]

        keys_to_change_outputs = [
                "model.diffusion_model.out.0.weight",
                "model.diffusion_model.out.0.bias",
                "model.diffusion_model.out.2.weight",
                "model.diffusion_model.out.2.bias",
                "model_ema.diffusion_modelout0weight",
                "model_ema.diffusion_modelout0bias",
                "model_ema.diffusion_modelout2weight",
                "model_ema.diffusion_modelout2bias",
        ]

        rank_zero_print("Modifying weights and biases to multiply their number of output channels")
        #initializes randomly new output weights to match new implementation
        for k in range(len(keys_to_change_outputs)):
            key_old = keys_to_change_outputs[k]
            key_new = keys_to_change_outputs_new[k]
            print("modifying output weights for compatibility")
            output_weight_new = new_state[key_new] # size (C_out, C_in, kernel_size, kernel_size)
            C_out_new = output_weight_new.size(0)
            C_out_old = old_state[key_old].size(0)

            assert C_out_new % C_out_old == 0 and  C_out_new >= C_out_old, f"Number of input channels for checkpoint and new U-Net should be multiple, got {C_out_new} and {C_out_old}"
            print("modifying input weights for compatibility")
            #repeats checkpoint weights C_in_new//C_in_old times along output weights channels=dim0
            copy_weights = True
            scale = 1 if copy_weights else 1e-8 #copies exactly weights if copy initialization
            old_state[key_new] = modify_conv_weights(old_state[key_old], scale=scale, n=C_out_new//C_out_old, dim=0, copy_weights=copy_weights)

        #if we load SD weights and still want to override with existing checkpoints
        if config.model.params.ckpt_path is not None:
            rank_zero_print(f"Attempting to load state from {config.model.params.ckpt_path}")
            old_state = torch.load(config.model.params.ckpt_path, map_location="cpu")
            if "state_dict" in old_state:
                rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]
            #loads checkpoint weights
            m, u = model.load_state_dict(old_state, strict=False)
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

        #loads checkpoint weights
        m, u = model.load_state_dict(old_state, strict=False)
        if len(m) > 0:
            rank_zero_print("missing keys:")
            rank_zero_print(m)
        if len(u) > 0:
            rank_zero_print("unexpected keys:")
            rank_zero_print(u)

        #if we load SD weights and still want to override with existing checkpoints
        if config.model.params.ckpt_path is not None:
            rank_zero_print(f"Attempting to load state from {config.model.params.ckpt_path}")
            old_state = torch.load(config.model.params.ckpt_path, map_location="cpu")
            if "state_dict" in old_state:
                rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
                old_state = old_state["state_dict"]
            #loads checkpoint weights
            m, u = model.load_state_dict(old_state, strict=False)
            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)


    ################## jg: end main trunk, start sampling specifics ####################
    # TODO - remove trainer config
    assert len(config.lightning.trainer.gpus) == 1, f"Sample on a single GPU, {len(config.lightning.trainer.gpus)} are provided"
    gpu = config.lightning.trainer.gpus[0]
    device = torch.device(f'cuda:{gpu}')
    model.eval()
    model.to(device)

    # Creates datamodule, dataloader, dataset
    assert len(config.data.params.validation.params.val_scenes) == 1, "Provide a single validation scene for sampling."
    scene = config.data.params.validation.params.val_scenes[0]
    dataset_name = Path(config.data.params.validation.params.root_dir).name
    data = instantiate_from_config(config.data) # cree le datamodule
    data.prepare_data()
    data.setup() #cree les datasets
    dataloader_val = data.val_dataloader()
    # dataset_train = data.datasets['train']

    # Visualization paths.
    viz_path_videos = Path(os.path.join('visualizations/medias/sampling', dataset_name, scene, 'modalities'))
    viz_path_pc= Path(os.path.join('visualizations/pointclouds', dataset_name, scene))
    os.makedirs(viz_path_videos, exist_ok=True)
    os.makedirs(viz_path_pc, exist_ok=True)

    depths_flowmap = dict()
    points_visualization = dict()
    logs = dict()

    # Sampling loop.
    for i, batch in tqdm(enumerate(dataloader_val)):
        
        print(f'Frame indices val batch {i} : ', batch['indices'])

        # Sample model.
        log_image_kwargs = config.lightning.callbacks.image_logger.params.log_images_kwargs
        batch_logs = model.log_images(
            batch,
            **log_image_kwargs
        )

        # Get flowmap inputs.
        if 'depth_ctxt' in batch_logs and 'depth_trgt' in batch_logs:
            x, c, flows_input, _  = model.get_input(batch, model.first_stage_key, return_flows_depths=True)

            bs, H, W = batch_logs['depth_ctxt']['samples'].squeeze(1).size()

            # Add gt and sampled depth.
            default_log(logs_dict=logs, modality='depths', batch_logs=batch_logs)
            
            # Add gt flow.
            default_add(logs, ['gt','flows','backward'], flows_input.backward.squeeze(1).cpu() * 0.0213) # (frame h w xy=2)
            default_add(logs, ['gt','flows','forward_mask'], flows_input.forward_mask.squeeze(1).cpu()) # (frame h w)
            default_add(logs, ['gt','flows','backward_mask'], flows_input.backward_mask.squeeze(1).cpu())

            # Add gt camera intrinsics.
            default_add(logs,['gt','ctxt','cameras'], batch['camera_ctxt'], operation="extend")
            default_add(logs,['gt','trgt','cameras'], batch['camera_trgt'], operation="extend")

        # Add gt and sampled flows and rgbs.
        default_log(logs_dict=logs, modality='rgbs', batch_logs=batch_logs)
        default_log(logs_dict=logs, modality='flows', batch_logs=batch_logs)
            

    # print_tensor_dict(logs)

    # Concatenates all tensors.
    for modality in ['rgbs', 'depths', 'flows']:
        log_dict_keys_list, _ = get_keys(modality)
        for keys in log_dict_keys_list:
            value = get_value(logs, keys)
            if value is not None:
                print(keys, type(value))
                value = torch.cat(value, dim=0)
                default_set(dictionnary=logs, keys=keys, value=value)
                print(keys, get_value(logs, keys).shape)
                print('')

    # Run flowmap for 3D projection.
    if config.experiment_cfg.save_points:
        assert 'depth_ctxt' in batch_logs and 'depth_trgt' in batch_logs
        
        for key in ['gt', 'sampled']:
            flows_flowmap = dict()

            # Create depths and correspondence weights inputs.
            depths_flowmap = torch.stack([
                get_value(logs, [key, 'ctxt', 'depths']),
                get_value(logs, [key, 'trgt', 'depths'])],
                dim=0
            ).to(model.device) # (batch=2 frame height width) batch[0] = ctxt, batch[1] = trgt
            
            # ############# TODO remove depth filtering #################
            depths_flowmap, valid_depths = filter_depth(depths_flowmap)

            valid_depths = valid_depths.float().to(model.device)
            # TODO - to fix
            correspondence_weights_flowmap = valid_depths[:,:-1,:,:] * valid_depths[:,1:,:,:] # both depth points should be valid
            default_set(logs, [key,'ctxt','correspondence_weights'], correspondence_weights_flowmap[0].cpu())
            default_set(logs, [key,'trgt','correspondence_weights'], correspondence_weights_flowmap[1].cpu())

            # Create depths and correspondence weights inputs.
            # correspondence_weights_flowmap = torch.stack([
            #     get_value(logs, [key,'ctxt','correspondence_weights']),
            #     get_value(logs, [key,'trgt','correspondence_weights'])],
            #     dim=0
            # ).to(model.device) # (batch=2 pair=frame-1 height width) batch[0] = ctxt, batch[1] = trgt
            # depths_flowmap = torch.stack([
            #     get_value(logs, [key, 'ctxt', 'depths']),
            #     get_value(logs, [key, 'trgt', 'depths'])],
            #     dim=0
            # ).to(model.device) # (batch=2 frame height width) batch[0] = ctxt, batch[1] = trgt

            # Create flows inputs.
            for keys in ['forward', 'backward','forward_mask','backward_mask']:
                flows_flowmap[keys] = torch.stack([
                    get_value(logs, ['gt','flows',keys])[:-1],
                    get_value(logs, ['gt','flows',keys])[1:]],
                    dim=0
                ).to(model.device)
            flows_flowmap = Flows(**flows_flowmap)

            # Create a dummy video batch.
            B, F, H, W = depths_flowmap.shape
            P_flow = flows_flowmap.forward.size(1)
            P_corr = correspondence_weights_flowmap.size(1)
            assert F - 1 == P_flow == P_corr
            dummy_flowmap_batch = {
                "videos": torch.zeros((B, F, 3, H, W), dtype=torch.float32, device=model.device),
                "indices": torch.stack([
                        torch.arange(F, device=model.device),
                        torch.arange(1, F+1, device=model.device)],
                    dim=0
                ),
                "scenes": [""],
                "datasets": [""],
            }
            if 'camera_ctxt' in batch and 'camera_trgt' in batch:
                intrinsics_flowmap = torch.stack([
                    torch.stack([cam.K for cam in get_value(logs, ['gt','ctxt','cameras'])], dim=0).float(),
                    torch.stack([cam.K for cam in get_value(logs, ['gt','trgt','cameras'])], dim=0).float()],
                    dim=0
                ).to(model.device) # (batch=2 frame 3 3) batch[0] = ctxt, batch[1] = trgt
                intrinsics_flowmap[:,:,:2,:3] = intrinsics_flowmap[:,:,:2,:3] / H
                dummy_flowmap_batch['intrinsics'] = intrinsics_flowmap

            # Compute Flowmap output for full sequence.
            _, flowmap_output = model.flowmap_loss_wrapper(
                dummy_flowmap_batch,
                flows_flowmap,
                depths_flowmap,
                correspondence_weights_flowmap,
                model.global_step,
                return_outputs=True
            )

            # Convert depths to world points.


            # ############### David ##############
            # from ldm.modules.flowmap.model.projection import sample_image_grid
            # from ldm.modules.flowmap.model.projection import unproject, homogenize_points
            # from einops import einsum
            # _, _, dh, dw = flowmap_output.depths.shape
            # xy, _ = sample_image_grid((dh, dw), flowmap_output.extrinsics.device)
            
            # bundle_ctxt = zip(
            #     flowmap_output.extrinsics[0],
            #     flowmap_output.intrinsics[0],
            #     flowmap_output.depths[0],
            # )
            # points_ctxt = []
            # for extrinsics, intrinsics, depths in bundle_ctxt:
            #     xyz = unproject(xy, depths, intrinsics)
            #     xyz = homogenize_points(xyz)
            #     xyz = einsum(extrinsics, xyz, "i j, ... j -> ... i")[..., :3]
            #     points_ctxt.append(rearrange(xyz, "h w xyz -> (h w) xyz").detach().cpu().numpy())
            # points_ctxt = np.stack(points_ctxt, axis=0)

            # bundle_trgt = zip(
            #     flowmap_output.extrinsics[1],
            #     flowmap_output.intrinsics[1],
            #     flowmap_output.depths[1],
            # )
            # points_trgt = []
            # for extrinsics, intrinsics, depths in bundle_trgt:
            #     xyz = unproject(xy, depths, intrinsics)
            #     xyz = homogenize_points(xyz)
            #     xyz = einsum(extrinsics, xyz, "i j, ... j -> ... i")[..., :3]
            #     points_trgt.append(rearrange(xyz, "h w xyz -> (h w) xyz").detach().cpu().numpy())
            # points_trgt = np.stack(points_trgt, axis=0)


            # ############ My way - depth to world ##############
            # H, W = flowmap_output.depths.shape[-2:]
            # for k in range(flowmap_output.extrinsics.size(1)):
                
            #     # Create camera.
            #     intrinsics_ctxt = K_to_intrinsics(flowmap_output.intrinsics[0,k].cpu(), [H,W])
            #     intrinsics_trgt = K_to_intrinsics(flowmap_output.intrinsics[1,k].cpu(), [H,W])
            #     w2c_ctxt = torch.linalg.inv(flowmap_output.extrinsics[0,k].cpu())
            #     w2c_trgt = torch.linalg.inv(flowmap_output.extrinsics[1,k].cpu())
            #     extrinsics_ctxt = Extrinsics(**{'R': w2c_ctxt[:3,:3], 't': w2c_ctxt[:3,3]})
            #     extrinsics_trgt = Extrinsics(**{'R': w2c_trgt[:3,:3], 't': w2c_trgt[:3,3]})
            #     camera_ctxt = Camera(intrinsics_ctxt, extrinsics_ctxt)
            #     camera_trgt = Camera(intrinsics_trgt, extrinsics_trgt)
                
            #     # Project depth to world.
            #     HW = pixel_grid_coordinates(H, W)
            #     pixel_coordinates = rearrange(HW, 'h w c -> c (h w)')
            #     depth_ctxt = rearrange(flowmap_output.depths[0,k].cpu(), 'h w -> 1 (h w)')
            #     depth_trgt = rearrange(flowmap_output.depths[1,k].cpu(), 'h w -> 1 (h w)')

            #     # Un-project to world coordinates.
            #     world_pts_ctxt = camera_ctxt.pixel_to_world(pixel_coordinates, depths=depth_ctxt)
            #     world_pts_trgt = camera_trgt.pixel_to_world(pixel_coordinates, depths=depth_trgt)
            #     world_coordinates_ctxt = to_euclidean_space(world_pts_ctxt)
            #     world_coordinates_trgt = to_euclidean_space(world_pts_trgt)

            #     default_add(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(world_coordinates_ctxt, 'c n -> n c'))
            #     default_add(points_visualization, [key, 'trgt', 'world_crds'], rearrange(world_coordinates_trgt, 'c n -> n c'))

            #     ############### Debug ##############

            #     # # Debug - GT depths and poses.
            #     # camera_ctxt = get_value(logs, ['gt','ctxt','cameras'])[k]
            #     # camera_trgt = get_value(logs, ['gt','trgt','cameras'])[k]

            #     # # Project depth to world.
            #     # HW = pixel_grid_coordinates(H, W)
            #     # pixel_coordinates = rearrange(HW, 'h w c -> c (h w)')
            #     # depth_ctxt = rearrange(get_value(logs, ['gt','ctxt','depths'])[k], 'h w -> 1 (h w)')
            #     # depth_trgt = rearrange(get_value(logs, ['gt','trgt','depths'])[k], 'h w -> 1 (h w)')

            #     # # Un-project to world coordinates.
            #     # world_pts_ctxt = camera_ctxt.pixel_to_world(pixel_coordinates, depths=depth_ctxt)
            #     # world_pts_trgt = camera_trgt.pixel_to_world(pixel_coordinates, depths=depth_trgt)
            #     # world_coordinates_ctxt = to_euclidean_space(world_pts_ctxt)
            #     # world_coordinates_trgt = to_euclidean_space(world_pts_trgt)

            #     # default_add(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(world_coordinates_ctxt, 'c n -> n c'))
            #     # default_add(points_visualization, [key, 'trgt', 'world_crds'], rearrange(world_coordinates_trgt, 'c n -> n c'))

                
            #     # #for debug only
            #     # default_add(points_visualization, [key, 'ctxt', 'poses'], camera_ctxt.c2w)
            #     # default_add(points_visualization, [key, 'trgt', 'poses'], camera_trgt.c2w)


        
            ############ My way - local to world ##############
            H, W = flowmap_output.depths.shape[-2:]
            for k in range(flowmap_output.extrinsics.size(1)):
                
                # Create camera.
                intrinsics_ctxt = K_to_intrinsics(flowmap_output.intrinsics[0,k].cpu(), [H,W])
                intrinsics_trgt = K_to_intrinsics(flowmap_output.intrinsics[1,k].cpu(), [H,W])
                w2c_ctxt = torch.linalg.inv(flowmap_output.extrinsics[0,k].cpu())
                w2c_trgt = torch.linalg.inv(flowmap_output.extrinsics[1,k].cpu())
                extrinsics_ctxt = Extrinsics(**{'R': w2c_ctxt[:3,:3], 't': w2c_ctxt[:3,3]})
                extrinsics_trgt = Extrinsics(**{'R': w2c_trgt[:3,:3], 't': w2c_trgt[:3,3]})
                camera_ctxt = Camera(intrinsics_ctxt, extrinsics_ctxt)
                camera_trgt = Camera(intrinsics_trgt, extrinsics_trgt)
                
                # Project local to world.
                local_pts_ctxt = to_projective_space(rearrange(flowmap_output.surfaces[0,k,:,:,:].cpu(), 'h w xyz -> xyz (h w)'))
                local_pts_trgt = to_projective_space(rearrange(flowmap_output.surfaces[1,k,:,:,:].cpu(), 'h w xyz -> xyz (h w)'))
                world_coordinates_ctxt = to_euclidean_space(camera_ctxt.local_to_world(local_pts_ctxt))
                world_coordinates_trgt = to_euclidean_space(camera_trgt.local_to_world(local_pts_trgt))

                default_add(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(world_coordinates_ctxt, 'c n -> n c'))
                default_add(points_visualization, [key, 'trgt', 'world_crds'], rearrange(world_coordinates_trgt, 'c n -> n c'))



            # # Store David.
            # default_set(points_visualization, [key, 'ctxt', 'world_crds'], points_ctxt)
            # default_set(points_visualization, [key, 'trgt', 'world_crds'], points_trgt)
            # default_set(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # default_set(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # default_set(points_visualization, [key,'ctxt', 'poses'], flowmap_output.extrinsics[0].cpu().numpy())
            # default_set(points_visualization, [key,'trgt', 'poses'], flowmap_output.extrinsics[1].cpu().numpy()) # cam-to-world

            
            
            # Store my way.
            default_set(
                dictionnary=points_visualization,
                keys=[key,'ctxt','world_crds'],
                value=torch.stack(get_value(points_visualization, [key, 'ctxt', 'world_crds']), dim=0).numpy()
            )
            default_set(
                dictionnary=points_visualization,
                keys=[key,'trgt','world_crds'],
                value=torch.stack(get_value(points_visualization, [key,'trgt','world_crds']), dim=0).numpy()
            ) # (frame, hw, xyz=3)
            default_set(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            default_set(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            default_set(points_visualization, [key,'ctxt', 'poses'], flowmap_output.extrinsics[0].cpu().numpy())
            default_set(points_visualization, [key,'trgt', 'poses'], flowmap_output.extrinsics[1].cpu().numpy()) # cam-to-world


            # # Store debug.
            # default_set(
            #     dictionnary=points_visualization,
            #     keys=[key,'ctxt','world_crds'],
            #     value=torch.stack(get_value(points_visualization, [key, 'ctxt', 'world_crds']), dim=0).numpy()
            # )
            # default_set(
            #     dictionnary=points_visualization,
            #     keys=[key,'trgt','world_crds'],
            #     value=torch.stack(get_value(points_visualization, [key,'trgt','world_crds']), dim=0).numpy()
            # ) # (frame, hw, xyz=3)
            # default_set(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # default_set(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # default_set(
            #     dictionnary=points_visualization,
            #     keys=[key,'ctxt','poses'],
            #     value=torch.stack(get_value(points_visualization, [key,'ctxt','poses']), dim=0).numpy()
            # )
            # default_set(
            #     dictionnary=points_visualization,
            #     keys=[key,'trgt','poses'],
            #     value=torch.stack(get_value(points_visualization, [key,'trgt','poses']), dim=0).numpy()
            # )


            # # Store old way (surfaces).
            # default_set(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(flowmap_output.surfaces[0], 'f h w xyz -> f (h w) xyz').cpu().numpy())
            # default_set(points_visualization, [key, 'trgt', 'world_crds'], rearrange(flowmap_output.surfaces[1], 'f h w xyz -> f (h w) xyz').cpu().numpy())
            # default_set(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # default_set(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # default_set(points_visualization, [key,'ctxt', 'poses'], flowmap_output.extrinsics[0].cpu().numpy())
            # default_set(points_visualization, [key,'trgt', 'poses'], flowmap_output.extrinsics[1].cpu().numpy()) # cam-to-world
            # # flowmap_points = flowmap_output.surfaces #(batch frame height width xyz=3)
            # # flowmap_intrinsics = flowmap_output.intrinsics #(batch frame 3 3)
            # # flowmap_extrinsics = flowmap_output.extrinsics #(batch frame 4 4)
            # # flowmap_depths = flowmap_output.depths #(batch frame height width) = depths


        # print_tensor_dict(points_visualization)

        np.savez(os.path.join(viz_path_pc, f'{dataset_name}_{scene}_.npz'), **points_visualization)

    # Save video visualizations.
    if config.experiment_cfg.visualization:
        viz_dict = dict()

        # Save RGB, depth, flow samples.
        for modality in ['rgbs', 'depths', 'flows']:
            keys_list, _ = get_keys(modality)
            for keys in keys_list:
                value = get_value(logs, keys)
                if value is not None:
                    viz_images = prepare_visualization(value, keys)
                    default_set(viz_dict, keys, viz_images)
                    video = (viz_images * 255).type(torch.uint8)
                    torchvision.io.write_video(os.path.join(viz_path_videos, f'{"_".join(keys)}.mp4'), video, fps=5)
        # Save correspondence masks samples.
        for keys in product(['gt', 'sampled'], ['ctxt', 'trgt'], ['correspondence_weights']):
            value = get_value(logs, keys)
            if value is not None:
                viz_images = prepare_visualization(value, keys)
                video = (viz_images * 255).type(torch.uint8)
                torchvision.io.write_video(os.path.join(viz_path_videos, f'{"_".join(keys)}.mp4'), video, fps=5)
        try:
            from visualizations.code.create_figure_sampling import create_video
            create_video(viz_dict, viz_path_videos.parent)
            # os.system(f'python -m visualizations.code.create_figure_sampling {viz_path_videos.parent}')
        except ModuleNotFoundError:
            pass

        # Compute consistency mask.
        for key in ['gt', 'sampled']:
            fwd_flows_gt = get_value(logs, [key, 'flows', 'forward'])
            src_im_gt = rearrange(get_value(logs, ['gt', 'ctxt', 'rgbs']), 'n h w c -> n c h w')
            trgt_im_gt = rearrange(get_value(logs, ['gt', 'trgt', 'rgbs']), 'n h w c -> n c h w')
            masks = torch.stack(
                [compute_consistency_mask(src_im_gt[k], trgt_im_gt[k], fwd_flows_gt[k]) for k in range(fwd_flows_gt.size(0))],
                dim = 0
            )

            video = (repeat(masks, 'n h w -> n h w c', c=3) * 255).type(torch.uint8)
            torchvision.io.write_video(os.path.join(viz_path_videos, f'{key}_consistency_masks.mp4'), video, fps=5)

            
if __name__ == "__main__":
    sample()
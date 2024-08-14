import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import hydra
import pytorch_lightning as pl
import json
import copy

from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from einops import rearrange, repeat
from torchvision.utils import save_image
from tqdm import tqdm
from pathlib import Path
from itertools import product
from contextlib import nullcontext
from torchvision.transforms import functional as FT

from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.misc.util import instantiate_from_config, modify_conv_weights, get_value, set_nested
from ldm.data.utils.camera import pixel_grid_coordinates, to_euclidean_space, to_projective_space, Camera, K_to_intrinsics, Extrinsics
from ldm.visualization import filter_depth
from ldm.thirdp.flowmap.flowmap.flow.flow_predictor import Flows
from ldm.misc.projection import compute_flow_projection, compute_consistency_mask
from ldm.misc.modalities import Modality, Modalities
from ldm.visualization.utils import *

# Enable arithmetic operations in .yaml file with keywords "divide" or "multiply" or "linear".
OmegaConf.register_new_resolver("divide", (lambda x, y: x//y), replace=True)
OmegaConf.register_new_resolver("multiply", (lambda x, y: x*y), replace=True)
OmegaConf.register_new_resolver("linear", (lambda a, b, c: a * b + c), replace=True)

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
            else:
                tmp_dict[k] = len(v)

    if root:
        print(json.dumps(tmp_dict, indent=4))


################### New logging functions ###################


def prepare_batch(
        modalities_cond: Modalities,
        modalities_out: Modalities,
        logs_data_dict: dict,
        batch: dict,
    ) -> dict:
    '''Put last RGB sample in input batch for autoregressive sampling.
    '''
    # modalities = Modalities(modalities_list)
    batch_for_model = copy.deepcopy(batch)
    prev_sample = get_value(logs_data_dict, ['rgb', 'trgt', 'sample']) # list | None
    prev_conditioning = get_value(logs_data_dict, ['rgb', 'ctxt', 'conditioning']) # list | None

    if not (modalities_cond.ids == ['ctxt_rgb'] and 'trgt_rgb' in modalities_out.ids and prev_sample is not None and prev_conditioning is not None and 'ctxt_rgb' in batch.keys()):
        raise NotImplementedError('Autoregressive sampling only implemented for RGB conditioning.')
    
    # Replace last sample in conditioning file.
    conditioning_rgb = torch.cat(
        [
            prev_conditioning[-1][:,1:],
            prev_sample[-1][:,:1]
        ],
        dim=1
    )

    conditioning_rgb = rearrange(conditioning_rgb, 'b f c h w -> b f h w c')
    batch_for_model['ctxt_rgb'] = conditioning_rgb

    return batch_for_model

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
        ckpt_cfg_files = sorted(glob.glob(os.path.join(ckpt_cfg_path, "*project.yaml")))
        # print(0.3, ckpt_cfg_files)
        # assert len(ckpt_cfg_files) == 1
        cfg_file = str(ckpt_cfg_files[-1])
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
    log_images_kwargs = dict(config.lightning.callbacks.image_logger.params.log_images_kwargs)

    # TODO - remove trainer config
    assert len(config.lightning.trainer.gpus) == 1, f"Sample on a single GPU, {len(config.lightning.trainer.gpus)} are provided"
    gpu = config.lightning.trainer.gpus[0]
    device = torch.device(f'cuda:{gpu}')
    model.eval()
    model.to(device)

    # Creates datamodule, dataloader, dataset
    assert len(config.data.params.validation.params.val_scenes) == 1, "Provide a single validation scene for sampling."
    config.data.params.batch_size = 1
    scene = config.data.params.validation.params.val_scenes[0]
    dataset_name = Path(config.data.params.validation.params.root_dir).name
    data = instantiate_from_config(config.data) # cree le datamodule
    data.prepare_data()
    data.setup() #cree les datasets
    dataloader_val = data.val_dataloader()

    # Visualization paths.
    viz_path_videos = Path(os.path.join('visualizations/medias/sampling', dataset_name, scene, 'modalities'))
    viz_path_pc = Path(os.path.join('visualizations/pointclouds', dataset_name, scene))
    os.makedirs(viz_path_videos, exist_ok=True)
    os.makedirs(viz_path_pc, exist_ok=True)

    depths_flowmap = dict()
    points_visualization = dict()
    logs_data = dict()
    logs_vis = dict()
    modalities = list()

    ema_scope = model.ema_scope if log_images_kwargs.get('use_ema_scope', False) else nullcontext
    use_ddim = log_images_kwargs.get('ddim_steps') is not None and model.num_timesteps > 1
    ddim_eta = log_images_kwargs.get('ddim_eta', 1.)
    ddim_steps = log_images_kwargs.get('ddim_steps')

    if model.n_future > 1:
        raise NotImplemented('Sampling not implented for n_future > 1 (in particular autoregressive).')

    # Sampling loop.
    for i, batch in tqdm(enumerate(dataloader_val)):

        if i % config.data.params.validation.params.stride == 0: # TODO - handle so that it actually corresponds to index even when bs neq 1
            print(f'Frame indices val batch {i} : ', batch['indices'])

            # Prepare input batch.
            if i > 0 and config.experiment_cfg.autoregressive:
                batch_for_model = prepare_batch(
                    model.modalities_cond,
                    model.modalities_out,
                    logs_data,
                    batch,
                )
            else:
                batch_for_model = batch

            # Get model inputs
            x, c, xc = model.get_input(
                batch_for_model,
                force_c_encode=True,
                return_original_cond=True,
            )
            x_log = rearrange(x, 'b (f c) h w -> b f c h w', c=model.channels_m)
            xc_log = rearrange(xc, 'b (f c) h w -> b f c h w', c=model.channels_m)
            x_split = model.modalities_in.split_modalities_multiplicity(x_log)
            xc_split = model.modalities_cond.split_modalities_multiplicity(xc_log)

            # Sample model.
            with ema_scope("Sampling"):
                samples, x_denoise_row = model.sample_log(
                    cond=c,
                    batch_size=x.size(0),
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta
                )
            weights = samples.weights
            samples_diffusion = rearrange(samples.x_denoised, 'b (f c) h w -> b f c h w', c=model.channels_m)
            samples_prediction = rearrange(samples.x_predicted, 'b (f c) h w -> b f c h w', c=model.channels_m)
            samples_diffusion = model.modalities_out.split_modalities_multiplicity(samples_diffusion, modality_ids=model.modalities_out.ids_denoised)
            samples_prediction = model.modalities_out.split_modalities_multiplicity(samples_prediction, modality_ids=model.modalities_out.ids_clean)
            samples = dict(samples_diffusion, **samples_prediction)

            # Prepare and log conditioning visualizations and raw data.
            for modality in model.modalities_cond:
                if not modality in model.modalities_in:
                    modalities.append(modality)
                    x_m = xc_split[modality._id].detach().cpu().clone()
                    kwargs = copy.deepcopy(log_images_kwargs)
                    visualization = prepare_visualization(x_m[:,-1:], modality, **kwargs) # TODO - remove hack, subsamples only last frame for viz / make grid
                    keys = [modality.modality, modality.name, 'conditioning']
                    set_nested(
                        logs_vis,
                        keys,
                        visualization
                    )
                    set_nested(
                        logs_data,
                        keys,
                        x_m
                    )
            
            # Log input visualizations and raw data.
            for modality in model.modalities_in:
                modalities.append(modality)
                x_m = x_split[modality._id].detach().cpu().clone()
                kwargs = copy.deepcopy(log_images_kwargs)
                visualization = prepare_visualization(x_m[:,-1:], modality, **kwargs) # TODO - remove hack, subsamples only last frame for viz / make grid
                keys = [modality.modality, modality.name, 'input']
                set_nested(
                    logs_vis,
                    keys,
                    visualization
                )
                set_nested(
                    logs_data,
                    keys,
                    x_m
                )

            # Log samples and gt and raw data.
            for modality in model.modalities_out:
                modalities.append(modality)
                sample_m = samples[modality._id].detach().cpu().clone()
                kwargs = copy.deepcopy(log_images_kwargs)
                kwargs['sample'] = True
                visualization = prepare_visualization(sample_m[:,-1:], modality, **kwargs) # TODO - remove hack, subsamples only last frame for viz / make grid
                keys = [modality.modality, modality.name, 'sample']
                set_nested(
                    logs_vis,
                    keys,
                    visualization
                )
                set_nested(
                    logs_data,
                    keys,
                    sample_m
                )

                if modality._id in batch.keys() and modality not in model.modalities_in: #gt
                    x_m = model.get_input_modality(batch, modality._id).detach().cpu().clone()
                    x_m = rearrange(x_m, 'b (f c) h w -> b f c h w', c=modality.channels_m)
                    kwargs = copy.deepcopy(log_images_kwargs)
                    visualization = prepare_visualization(x_m[:,-1:], modality, **kwargs) # TODO - remove hack, subsamples only last frame for viz / make grid
                    keys = [modality.modality, modality.name, 'gt']
                    set_nested(
                        logs_vis,
                        keys,
                        visualization
                    )
                    set_nested(
                        logs_data,
                        keys,
                        x_m
                    )

            # Log correspondence weight samples.
            if weights is not None:
                modality_weight = Modality(name='correspondence', modality='weight', multiplicity=weights.size(1), channels_m=0, denoised=False)
                modalities.append(modality_weight)
                weights = weights.detach().cpu().clone()
                kwargs = copy.deepcopy(log_images_kwargs)
                visualization = prepare_visualization(weights[:,-1:], modality_weight, **kwargs) # TODO - remove hack, subsamples only last frame for viz, concatenate everything / make grid
                keys = [modality_weight.modality, modality_weight.name, 'sample']
                set_nested(
                    logs_vis,
                    keys,
                    visualization
                )
                set_nested(
                    logs_data,
                    keys,
                    weights
                )

    # Concatenate all raw data.
    modalities = Modalities(modalities)
    for modality in modalities:
        keys_base = [modality.modality, modality.name]
        modality_logs_data = get_value(logs_data, keys_base)
        modality_logs_vis = get_value(logs_vis, keys_base)
        for key in modality_logs_data.keys():
            keys = keys_base + [key]
            modality_logs_data[key] = torch.cat(modality_logs_data[key], dim=0)
            modality_logs_vis[key] = torch.cat(modality_logs_vis[key], dim=0)

    # Run flowmap for 3D projection.
    if config.experiment_cfg.save_points:
        raise NotImplementedError('Flowmap 3D projection not implemented.')
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
            set_nested(logs, [key,'ctxt','correspondence_weights'], correspondence_weights_flowmap[0].cpu(), operation="set")
            set_nested(logs, [key,'trgt','correspondence_weights'], correspondence_weights_flowmap[1].cpu(), operation="set")

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
            # from ldm.thirdp.flowmap.flowmap.model.projection import sample_image_grid
            # from ldm.thirdp.flowmap.flowmap.model.projection import unproject, homogenize_points
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

            #     set_nested(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(world_coordinates_ctxt, 'c n -> n c'))
            #     set_nested(points_visualization, [key, 'trgt', 'world_crds'], rearrange(world_coordinates_trgt, 'c n -> n c'))

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

            #     # set_nested(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(world_coordinates_ctxt, 'c n -> n c'))
            #     # set_nested(points_visualization, [key, 'trgt', 'world_crds'], rearrange(world_coordinates_trgt, 'c n -> n c'))

                
            #     # #for debug only
            #     # set_nested(points_visualization, [key, 'ctxt', 'poses'], camera_ctxt.c2w)
            #     # set_nested(points_visualization, [key, 'trgt', 'poses'], camera_trgt.c2w)


        
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

                set_nested(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(world_coordinates_ctxt, 'c n -> n c'))
                set_nested(points_visualization, [key, 'trgt', 'world_crds'], rearrange(world_coordinates_trgt, 'c n -> n c'))



            # # Store David.
            # set_nested(points_visualization, [key, 'ctxt', 'world_crds'], points_ctxt, operation="set")
            # set_nested(points_visualization, [key, 'trgt', 'world_crds'], points_trgt, operation="set")
            # set_nested(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy(), operation="set")
            # set_nested(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy(), operation="set")
            # set_nested(points_visualization, [key,'ctxt', 'poses'], flowmap_output.extrinsics[0].cpu().numpy(), operation="set")
            # set_nested(points_visualization, [key,'trgt', 'poses'], flowmap_output.extrinsics[1].cpu().numpy(), operation="set") # cam-to-world

            
            
            # Store my way.
            set_nested(
                dictionnary=points_visualization,
                keys=[key,'ctxt','world_crds'],
                value=torch.stack(get_value(points_visualization, [key, 'ctxt', 'world_crds']), dim=0).numpy(),
                operation="set"
            )
            set_nested(
                dictionnary=points_visualization,
                keys=[key,'trgt','world_crds'],
                value=torch.stack(get_value(points_visualization, [key,'trgt','world_crds']), dim=0).numpy(),
                operation="set"
            ) # (frame, hw, xyz=3)
            set_nested(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy(), operation="set")
            set_nested(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy(), operation="set")
            set_nested(points_visualization, [key,'ctxt', 'poses'], flowmap_output.extrinsics[0].cpu().numpy(), operation="set")
            set_nested(points_visualization, [key,'trgt', 'poses'], flowmap_output.extrinsics[1].cpu().numpy(), operation="set") # cam-to-world


            # # Store debug.
            # set_nested(
            #     dictionnary=points_visualization,
            #     keys=[key,'ctxt','world_crds'],
            #     value=torch.stack(get_value(points_visualization, [key, 'ctxt', 'world_crds']), dim=0).numpy(),
            #     operation="set"

            # )
            # set_nested(
            #     dictionnary=points_visualization,
            #     keys=[key,'trgt','world_crds'],
            #     value=torch.stack(get_value(points_visualization, [key,'trgt','world_crds']), dim=0).numpy(),
            #     operation="set"
            # ) # (frame, hw, xyz=3)
            # set_nested(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # set_nested(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy())
            # set_nested(
            #     dictionnary=points_visualization,
            #     keys=[key,'ctxt','poses'],
            #     value=torch.stack(get_value(points_visualization, [key,'ctxt','poses']), dim=0).numpy(), operation="set"
            # )
            # set_nested(
            #     dictionnary=points_visualization,
            #     keys=[key,'trgt','poses'],
            #     value=torch.stack(get_value(points_visualization, [key,'trgt','poses']), dim=0).numpy(), operation="set"
            # )


            # # Store old way (surfaces).
            # set_nested(points_visualization, [key, 'ctxt', 'world_crds'], rearrange(flowmap_output.surfaces[0], 'f h w xyz -> f (h w) xyz').cpu().numpy(), operation="set")
            # set_nested(points_visualization, [key, 'trgt', 'world_crds'], rearrange(flowmap_output.surfaces[1], 'f h w xyz -> f (h w) xyz').cpu().numpy(), operation="set")
            # set_nested(points_visualization, [key,'ctxt', 'rgb_crds'], rearrange(get_value(logs, ['gt','ctxt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy(), operation="set")
            # set_nested(points_visualization, [key,'trgt', 'rgb_crds'], rearrange(get_value(logs, ['gt','trgt','rgbs']), 'f h w rgb -> f (h w) rgb').numpy(), operation="set")
            # set_nested(points_visualization, [key,'ctxt', 'poses'], flowmap_output.extrinsics[0].cpu().numpy())
            # set_nested(points_visualization, [key,'trgt', 'poses'], flowmap_output.extrinsics[1].cpu().numpy()) # cam-to-world
            # # flowmap_points = flowmap_output.surfaces #(batch frame height width xyz=3)
            # # flowmap_intrinsics = flowmap_output.intrinsics #(batch frame 3 3)
            # # flowmap_extrinsics = flowmap_output.extrinsics #(batch frame 4 4)
            # # flowmap_depths = flowmap_output.depths #(batch frame height width) = depths


        # print_tensor_dict(points_visualization)

        np.savez(os.path.join(viz_path_pc, f'{dataset_name}_{scene}_.npz'), **points_visualization)

    # Save video visualizations.
    if config.experiment_cfg.visualization:
        print(f'Saving videos at {viz_path_videos}.')

        for modality in modalities:
            keys_base = [modality.modality, modality.name]
            modality_logs_data = get_value(logs_data, keys_base)
            modality_logs_vis = get_value(logs_vis, keys_base)
            for key in modality_logs_data.keys():
                keys = keys_base + [key]
                frames = modality_logs_vis[key]

                # Resize to make sure height and width are cleanly divisible by two.
                new_shape = (torch.tensor(frames.shape[-2:]) // 2) * 2
                frames = FT.resize(frames, new_shape)
                frames = rearrange(frames, 'b c h w -> b h w c')
                frames = (frames * 255).type(torch.uint8)

                # Save video visualization.
                torchvision.io.write_video(
                    os.path.join(viz_path_videos, f'{"_".join(keys)}.mp4'),
                    frames,
                    fps=1
                )


        # # Save correspondence masks samples.
        # for keys in product(['gt', 'sampled'], ['ctxt', 'trgt'], ['correspondence_weights']):
        #     value = get_value(logs, keys)
        #     if value is not None:
        #         viz_images = prepare_visualization(value, keys)
        #         video = (viz_images * 255).type(torch.uint8)
        #         torchvision.io.write_video(os.path.join(viz_path_videos, f'{"_".join(keys)}.mp4'), video, fps=5)
        # try:
        #     from visualizations.code.create_figure_sampling import create_video
        #     create_video(viz_dict, viz_path_videos.parent)
        #     # os.system(f'python -m visualizations.code.create_figure_sampling {viz_path_videos.parent}')
        # except ModuleNotFoundError:
        #     pass

        # # Compute consistency mask.
        # for key in ['gt', 'sampled']:
        #     fwd_flows_gt = get_value(logs, [key, 'flows', 'forward'])
        #     src_im_gt = rearrange(get_value(logs, ['gt', 'ctxt', 'rgbs']), 'n h w c -> n c h w')
        #     trgt_im_gt = rearrange(get_value(logs, ['gt', 'trgt', 'rgbs']), 'n h w c -> n c h w')
        #     masks = torch.stack(
        #         [compute_consistency_mask(src_im_gt[k], trgt_im_gt[k], fwd_flows_gt[k]) for k in range(fwd_flows_gt.size(0))],
        #         dim = 0
        #     )

        #     video = (repeat(masks, 'n h w -> n h w c', c=3) * 255).type(torch.uint8)
        #     torchvision.io.write_video(os.path.join(viz_path_videos, f'{key}_consistency_masks.mp4'), video, fps=5)

            
if __name__ == "__main__":
    sample()
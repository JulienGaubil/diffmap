import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import hydra
import pytorch_lightning as pl

from packaging import version
from jaxtyping import Float
from torch import Tensor
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from einops import rearrange
from torchvision.utils import save_image
from tqdm import tqdm
from flow_vis_torch import flow_to_color
from pathlib import Path

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import instantiate_from_config, modify_conv_weights
from ldm.data.utils.camera import pixel_grid_coordinates, to_euclidean_space, to_projective_space
from ldm.modules.flowmap.visualization.depth import color_map_depth
from ldm.modules.flowmap.visualization.color import apply_color_map_to_image

MULTINODE_HACKS = False

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

# def get_parser(**parser_kwargs):
#     def str2bool(v):
#         if isinstance(v, bool):
#             return v
#         if v.lower() in ("yes", "true", "t", "y", "1"):
#             return True
#         elif v.lower() in ("no", "false", "f", "n", "0"):
#             return False
#         else:
#             raise argparse.ArgumentTypeError("Boolean value expected.")

#     parser = argparse.ArgumentParser(**parser_kwargs)
#     parser.add_argument(
#         "--finetune_from",
#         type=str,
#         nargs="?",
#         default="",
#         help="path to checkpoint to load model state from"
#     )
#     parser.add_argument(
#         "-n",
#         "--name",
#         type=str,
#         const=True,
#         default="",
#         nargs="?",
#         help="postfix for logdir",
#     )
#     parser.add_argument(
#         "-r",
#         "--resume",
#         type=str,
#         const=True,
#         default="",
#         nargs="?",
#         help="resume from logdir or checkpoint in logdir",
#     )
#     parser.add_argument(
#         "-b",
#         "--base",
#         nargs="*",
#         metavar="base_config.yaml",
#         help="paths to base configs. Loaded from left-to-right. "
#              "Parameters can be overwritten or added with command-line options of the form `--key value`.",
#         default=list(),
#     )
#     parser.add_argument(
#         "-t",
#         "--train",
#         type=str2bool,
#         const=True,
#         default=False,
#         nargs="?",
#         help="train",
#     )
#     parser.add_argument(
#         "--no-test",
#         type=str2bool,
#         const=True,
#         default=False,
#         nargs="?",
#         help="disable test",
#     )
#     parser.add_argument(
#         "-p",
#         "--project",
#         help="name of new or path to existing project"
#     )
#     parser.add_argument(
#         "-d",
#         "--debug",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=False,
#         help="enable post-mortem debugging",
#     )
#     parser.add_argument(
#         "-s",
#         "--seed",
#         type=int,
#         default=23,
#         help="seed for seed_everything",
#     )
#     parser.add_argument(
#         "-f",
#         "--postfix",
#         type=str,
#         default="",
#         help="post-postfix for default name",
#     )
#     parser.add_argument(
#         "-l",
#         "--logdir",
#         type=str,
#         default="logs",
#         help="directory for logging dat shit",
#     )
#     parser.add_argument(
#         "--scale_lr",
#         type=str2bool,
#         nargs="?",
#         const=True,
#         default=True,
#         help="scale base-lr by ngpu * batch_size * n_accumulate",
#     )
#     return parser


# def nondefault_trainer_args(opt):
#     parser = argparse.ArgumentParser()
#     parser = Trainer.add_argparse_args(parser)
#     args = parser.parse_args([])
#     return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))




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
    viz_path_videos = Path(os.path.join('visualizations/medias/sampling', dataset_name, scene))
    viz_path_pc= Path(os.path.join('visualizations/pointclouds', dataset_name, scene))
    os.makedirs(viz_path_videos, exist_ok=True)
    os.makedirs(viz_path_pc, exist_ok=True)

    out = {
        "flows": {'fwd_flow_gt': [], 'fwd_flow_sampled': []},
        "rgbs": {
            'ctxt_rgb_gt': [],
            'trgt_rgb_gt': [],
            'trgt_rgb_sampled': []
        },
        "depths": {
            'ctxt_depth_gt': [],
            'ctxt_depth_sampled': [],
            'trgt_depth_gt': [],
            'trgt_depth_sampled': [] 
        }
    }


    # Create output.
    out_ = dict()
    for k in ['gt', 'sampled']:
        out_[k] = dict()
        for frame in ['ctxt', 'trgt']:
            out_[k][frame] = dict()
            for modality in ['rgb', 'depth']:
                if not (k == 'sampled' and 'frame' == 'ctxt' and modality == 'rb'):
                    out_[k][frame][modality] = []
        out_[k]['optical_flow'] = []

    outputs = dict()
    cameras = []
    for i, batch in tqdm(enumerate(dataloader_val)):
        print(f'Frame indices val batch {i} : ', batch['indices'])
        ########### jg: New sampling ################

        cameras += [(batch['camera_ctxt'][k], batch['camera_trgt'][k]) for k in range(len(batch['camera_trgt']))]

        log_image_kwargs = config.lightning.callbacks.image_logger.params.log_images_kwargs
        log = model.log_images(
            batch,
            **log_image_kwargs
        )
        # print(1, log.keys())

        if i > 0:
            for modality, logs_modality in log.items():
                for key, data in logs_modality.items():
                    outputs[modality][key] = torch.cat([outputs[modality][key].cpu(), data.cpu()], dim=0)
        else:
            outputs = log
            

        # Log flows.
        out['flows']['fwd_flow_gt'].append(log['optical_flow']['inputs'].cpu())
        out['flows']['fwd_flow_sampled'].append(log['optical_flow']['samples'].cpu())
        
        #Log rgbs.
        out['rgbs']['ctxt_rgb_gt'].append(log['conditioning']['concat'].cpu())
        out['rgbs']['trgt_rgb_gt'].append(log['trgt']['inputs'].cpu())
        out['rgbs']['trgt_rgb_sampled'].append(log['trgt']['samples'].cpu())

            
        # Log depths.
        out['depths']['ctxt_depth_sampled'].append(log['depth_ctxt']['samples'].cpu())
        out['depths']['trgt_depth_sampled'].append(log['depth_trgt']['samples'].cpu())
        out['depths']['ctxt_depth_gt'].append(log['depth_ctxt']['inputs'].cpu())
        out['depths']['trgt_depth_gt'].append(log['depth_trgt']['inputs'].cpu())



    print('Shapes default collage')
    for modality, logs_modality in outputs.items():
        for k, data in logs_modality.items():
            print(modality, k, data.shape)

    print('')
    print('Manual shapes')

    for modality, logs_modality in out.items():
        for k, data in logs_modality.items():
            out[modality][k] = torch.cat(data, dim=0)
            print(modality, k, out[modality][k].shape)


    # Save point cloud
    if True:
    # if config.prepare_visualization:

        # Log rgbs and depths for point cloud visualization.
        output_pointcloud = {
            'sampled': {'ctxt': dict(), 'trgt': dict()},
            'gt': {'ctxt': dict(), 'trgt': dict()}
        }
        
        for modality, logs_modality in outputs.items():

            if modality != 'optical_flow':
                for key, data in logs_modality.items():
                    split = 'sampled' if key != 'inputs' else 'gt'
                    
                    if 'trgt' in modality:
                        frame = 'trgt'
                    else:
                        frame = 'ctxt'

                    assert False, "Key rgb_crds not in dict for GT"

                    # Formats depth.
                    if 'depth' in modality:
                        if key in ['inputs', 'samples']:
                            depths = data.mean(1)
                            HW = pixel_grid_coordinates(depths.size(1), depths.size(2))
                            pixel_coordinates = rearrange(HW, 'h w c -> c (h w)')

                            world_crds = list()
                            poses = list()

                            for k in range(data.size(0)):
                                camera = cameras[k][0] if frame == 'ctxt' else cameras[k][1]
                                depth = rearrange(depths[k], 'h w -> 1 (h w)')

                                # Un-project to world coordinates
                                world_pts = camera.pixel_to_world(pixel_coordinates, depths=depth)
                                world_coordinates = to_euclidean_space(world_pts).numpy()

                                world_crds.append(rearrange(world_coordinates, 'c n -> n c'))
                                c2w = camera.c2w.numpy()
                                poses.append(c2w)

                            # Stacks points and poses
                            world_crds = np.stack(world_crds, axis=0) # (frame, hw, xyz=3)
                            poses = np.stack(poses, axis=0) # (frame, 4, 4)

                            output_pointcloud[split][frame]['world_crds'] = world_crds
                            output_pointcloud[split][frame]['poses'] = poses

                    # Format rgbs.
                    elif 'intermediates' not in key:
                        rgbs = torch.clamp(data, -1., 1.)
                        rgbs = (rgbs + 1.0) / 2.0
                        rgbs = rearrange(rgbs, 'n c h w -> n (h w) c').numpy() # (frame, hw, rgb=3)
                        output_pointcloud[split][frame]['rgb_crds'] = rgbs

        np.savez(os.path.join(viz_path_pc, f'{dataset_name}_{scene}_.npz'), **output_pointcloud)





    # Create videos from samples.
    for modality, logs_modality in out.items():
        for k, data in logs_modality.items():

            if modality == 'depths':
                if 'gt' in k:
                    out[modality][k] = out[modality][k].mean(1, keepdims=True)
                out[modality][k] = color_map_depth(out[modality][k].squeeze(1))

            elif modality == 'rgbs':
                out[modality][k] = torch.clamp(out[modality][k], -1., 1.)
                out[modality][k] = (out[modality][k] + 1.0) / 2.0

            elif modality == 'flows':
                out[modality][k] = (flow_to_color(out[modality][k][:,:2,:,:]) / 255)

            # Save videos.
            frames = rearrange(out[modality][k], 'b c h w -> b h w c')
            frames = (frames * 255).type(torch.uint8)
            torchvision.io.write_video(viz_path_videos / f'{scene}_{k}.mp4', frames, fps=5)

    # Save point cloud
    if False:
    # if config.prepare_visualization:
        viz_file = {'gt': dict(), 'sampled': dict()}
        # for k in out['depths']:
        #     if 'gt' in k:




        np.savez(f"samples_{dataset_name}_{scene}", **out)

    

    ########### jg: Instantiate loggers from main trunk ################

        # # trainer and callbacks
        # trainer_kwargs = dict()

        # # default logger configs
        # default_logger_cfgs = {
        #     "wandb": {
        #         "target": "pytorch_lightning.loggers.WandbLogger",
        #         "params": {
        #             "name": nowname,
        #             "save_dir": logdir,
        #             "offline": config.experiment_cfg.debug,
        #             "id": nowname,
        #         }
        #     },
        #     "testtube": {
        #         "target": "pytorch_lightning.loggers.TestTubeLogger",
        #         "params": {
        #             "name": "testtube",
        #             "save_dir": logdir,
        #         }
        #     },
        # }
        # default_logger_cfg = default_logger_cfgs["testtube"]
        # if "logger" in config.lightning:
        #     logger_cfg = config.lightning.logger
        # else:
        #     logger_cfg = OmegaConf.create()
        # logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        # trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # # add callback which sets up log directory
        # default_callbacks_cfg = {
        #     "setup_callback": {
        #         "target": "main.SetupCallback",
        #         "params": {
        #             "resume": config.experiment_cfg.resume,
        #             "now": now,
        #             "logdir": logdir,
        #             "ckptdir": ckptdir,
        #             "cfgdir": cfgdir,
        #             "config": config,
        #             "lightning_config": config.lightning,
        #             "debug": config.experiment_cfg.debug,
        #         }
        #     },
        #     "image_logger": {
        #         "target": "main.ImageLogger",
        #         "params": {
        #             "batch_frequency": 750,
        #             "max_images": 4,
        #             "clamp": True
        #         }
        #     },
        #     "cuda_callback": {
        #         "target": "main.CUDACallback"
        #     },
        # }

        # if "callbacks" in config.lightning:
        #     callbacks_cfg = config.lightning.callbacks
        # else:
        #     callbacks_cfg = OmegaConf.create()

        # callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        # if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
        #     callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        # elif 'ignore_keys_callback' in callbacks_cfg:
        #     del callbacks_cfg['ignore_keys_callback']

        # trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # if not "plugins" in trainer_kwargs:
        #     trainer_kwargs["plugins"] = list()
     
   

if __name__ == "__main__":
    sample()





########### jg: Old sampling code ################



#     now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

#     # add cwd for convenience and to make classes in this file available when
#     # running as `python main.py`
#     # (in particular `ldm.data.datamodule.DataModuleFromConfig`)
#     sys.path.append(os.getcwd())

#     parser = get_parser()
#     parser = Trainer.add_argparse_args(parser)

#     opt, unknown = parser.parse_known_args()
#     if opt.name and opt.resume:
#         raise ValueError(
#             "-n/--name and -r/--resume cannot be specified both."
#             "If you want to resume training in a new log folder, "
#             "use -n/--name in combination with --resume_from_checkpoint"
#         )
#     if opt.resume:
#         if not os.path.exists(opt.resume):
#             raise ValueError("Cannot find {}".format(opt.resume))
#         if os.path.isfile(opt.resume):
#             paths = opt.resume.split("/")
#             # idx = len(paths)-paths[::-1].index("logs")+1
#             # logdir = "/".join(paths[:idx])
#             logdir = "/".join(paths[:-2])
#             ckpt = opt.resume
#         else:
#             assert os.path.isdir(opt.resume), opt.resume
#             logdir = opt.resume.rstrip("/")
#             ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

#         opt.resume_from_checkpoint = ckpt
#         base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
#         opt.base = base_configs + opt.base
#         _tmp = logdir.split("/")
#         nowname = _tmp[-1]
#     else:
#         if opt.name:
#             name = "_" + opt.name
#         elif opt.base:
#             cfg_fname = os.path.split(opt.base[0])[-1]
#             cfg_name = os.path.splitext(cfg_fname)[0]
#             name = "_" + cfg_name
#         else:
#             name = ""
#         nowname = now + name + opt.postfix
#         logdir = os.path.join(opt.logdir, nowname)

#     ckptdir = os.path.join(logdir, "checkpoints")
#     cfgdir = os.path.join(logdir, "configs")
#     seed_everything(opt.seed)


#     # init and save configs
#     configs = [OmegaConf.load(cfg) for cfg in opt.base]
#     cli = OmegaConf.from_dotlist(unknown)
#     config = OmegaConf.merge(*configs, cli)
#     lightning_config = config.pop("lightning", OmegaConf.create())
#     # merge trainer cli with config
#     trainer_config = lightning_config.get("trainer", OmegaConf.create())
#     # default to ddp
#     trainer_config["accelerator"] = "ddp"
#     for k in nondefault_trainer_args(opt):
#         trainer_config[k] = getattr(opt, k)
#     if not "gpus" in trainer_config:
#         del trainer_config["accelerator"]
#         cpu = True
#     else:
#         gpuinfo = trainer_config["gpus"]
#         rank_zero_print(f"Running on GPUs {gpuinfo}")
#         device = torch.device(f'cuda:{gpuinfo}')
#         cpu = False
#     trainer_opt = argparse.Namespace(**trainer_config)
#     lightning_config.trainer = trainer_config

#     # model
#     model = instantiate_from_config(config.model)
#     model.cpu()

#     if not opt.finetune_from == "":
#         rank_zero_print(f"Attempting to load state from {opt.finetune_from}")
#         old_state = torch.load(opt.finetune_from, map_location="cpu")
#         if "state_dict" in old_state:
#             rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
#             old_state = old_state["state_dict"]

#         #Check if we need to port input weights from 4ch input to n*4ch
#         in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
#         new_state = model.state_dict()
#         in_filters_current = new_state["model.diffusion_model.input_blocks.0.0.weight"]
#         in_shape = in_filters_current.shape
#         #initializes new input weights if input dims don't match
#         if in_shape != in_filters_load.shape:
#             rank_zero_print("Modifying weights to multiply their number of input channels")
#             keys_to_change_inputs = [
#                 "model.diffusion_model.input_blocks.0.0.weight",
#                 "model_ema.diffusion_modelinput_blocks00weight",
#             ]
            
#             for k in keys_to_change_inputs:
#                 input_weight_new = new_state[k] # size (C_out, C_in, kernel_size, kernel_size)
#                 C_in_new = input_weight_new.size(1)
#                 C_in_old = old_state[k].size(1)

#                 assert  C_in_new % C_in_old == 0 and  C_in_new >= C_in_old, f"Number of input channels for checkpoint and new U-Net should be multiple, got {C_in_new} and {C_in_old}"
#                 print("modifying input weights for compatibitlity")
#                 copy_weights = True
#                 scale = 1/(C_in_new//C_in_old) if copy_weights else 1e-8 #scales input to prevent activations to blow up when copying
#                 #repeats checkpoint weights C_in_new//C_in_old times along input channels=dim1
#                 old_state[k] = modify_conv_weights(old_state[k], scale=scale, n=C_in_new//C_in_old, dim=1, copy_weights=copy_weights)
                
#         #check if we need to port output weights from 4ch to n*4 ch
#         out_filters_load = old_state["model.diffusion_model.out.2.weight"]
#         out_filters_current = new_state["model.diffusion_model.out.2.weight"]
#         out_shape = out_filters_current.shape
#         if out_shape != out_filters_load.shape:
#             rank_zero_print("Modifying weights and biases to multiply their number of output channels")
#             keys_to_change_outputs = [
#                 "model.diffusion_model.out.2.weight",
#                 "model.diffusion_model.out.2.bias",
#                 "model_ema.diffusion_modelout2weight",
#                 "model_ema.diffusion_modelout2bias",
#             ]
#             #initializes randomly new output weights if input dims don't match
#             for k in keys_to_change_outputs:
#                 print("modifying output weights for compatibitlity")
#                 output_weight_new = new_state[k] # size (C_out, C_in, kernel_size, kernel_size)
#                 C_out_new = output_weight_new.size(0)
#                 C_out_old = old_state[k].size(0)

#                 assert C_out_new % C_out_old == 0 and  C_out_new >= C_out_old, f"Number of input channels for checkpoint and new U-Net should be multiple, got {C_out_new} and {C_out_old}"
#                 print("modifying input weights for compatibitlity")
#                 #repeats checkpoint weights C_in_new//C_in_old times along output weights channels=dim0
#                 copy_weights = True
#                 scale = 1 if copy_weights else 1e-8 #copies exactly weights if copy initialization
#                 old_state[k] = modify_conv_weights(old_state[k], scale=scale, n=C_out_new//C_out_old, dim=0, copy_weights=copy_weights)

#         #if we load SD weights and still want to override with existing checkpoints
#         if config.model.params.ckpt_path is not None:
#             rank_zero_print(f"Attempting to load state from {config.model.params.ckpt_path}")
#             old_state = torch.load(config.model.params.ckpt_path, map_location="cpu")
#             if "state_dict" in old_state:
#                 rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
#                 old_state = old_state["state_dict"]
#             #loads checkpoint weights
#             m, u = model.load_state_dict(old_state, strict=False)
#             if len(m) > 0:
#                 rank_zero_print("missing keys:")
#                 rank_zero_print(m)
#             if len(u) > 0:
#                 rank_zero_print("unexpected keys:")
#                 rank_zero_print(u)

#         #loads checkpoint weights
#         m, u = model.load_state_dict(old_state, strict=False)
#         if len(m) > 0:
#             rank_zero_print("missing keys:")
#             rank_zero_print(m)
#         if len(u) > 0:
#             rank_zero_print("unexpected keys:")
#             rank_zero_print(u)

#         #if we load SD weights and still want to override with existing checkpoints
#         if config.model.params.ckpt_path is not None:
#             rank_zero_print(f"Attempting to load state from {config.model.params.ckpt_path}")
#             old_state = torch.load(config.model.params.ckpt_path, map_location="cpu")
#             if "state_dict" in old_state:
#                 rank_zero_print(f"Found nested key 'state_dict' in checkpoint, loading this instead")
#                 old_state = old_state["state_dict"]
#             #loads checkpoint weights
#             m, u = model.load_state_dict(old_state, strict=False)
#             if len(m) > 0:
#                 rank_zero_print("missing keys:")
#                 rank_zero_print(m)
#             if len(u) > 0:
#                 rank_zero_print("unexpected keys:")
#                 rank_zero_print(u)

#     model.eval()
#     model.to(device)

#     #creates datamodule, dataloader, dataset
#     data = instantiate_from_config(config.data) # cree le datamodule
#     data.prepare_data()
#     data.setup() #cree les datasets
#     dataloader_train = data.train_dataloader()
#     dataset_train = data.datasets['train']

#     bs = 4
#     N = 4
#     ddim_steps = 200
#     ddim_eta = 1.


#     flow_fwd_sampled_path = f"flow_forward_sampled"
#     flow_fwd_gt_path = f"flow_forward_gt"
#     os.makedirs(flow_fwd_sampled_path, exist_ok=True)
#     os.makedirs(flow_fwd_gt_path, exist_ok=True)

#     flows_fwd, flows_fwd_gt = list(), list()

#     assert model.modalities_in == ['optical_flow'], "Sampling not implemented for other modalities than optical flow"

#     for i, batch in tqdm(enumerate(dataloader_train)):
#         print('INDICES BATCH LOOP : ', batch['indices'])
#         x, c, xc = model.get_input(batch, model.first_stage_key,
#                                            force_c_encode=True,
#                                            return_original_cond=True,
#                                            bs=bs)
        
#         # Compute optical flow.
#         samples, x_denoise_row = model.sample_log(cond=c,batch_size=N,ddim=True,
#                                                             ddim_steps=ddim_steps,eta=ddim_eta) #samples generative process in latent space
        
#         # Prepare flow for logging.
#         samples_flow_fwd = samples.x_noisy.cpu()[1,:2,:,:].unsqueeze(0) # (N, C, H, W)
#         flow_fwd_gt = x.cpu()[1,:2,:,:].unsqueeze(0)  # (N, H, W, C)

#         # Saves flows, frames and masks.
#         flows_fwd.append(samples_flow_fwd)
#         flows_fwd_gt.append(flow_fwd_gt)
        
#         # Save RGB flow viz and frames.
#         flow_sampled_rgb = (flow_to_color(samples_flow_fwd) / 255)  #should be (B, 2, H, W)
#         flow_fwd_gt_rgb = (flow_to_color(flow_fwd_gt) / 255)

#         save_image(flow_sampled_rgb, os.path.join(flow_fwd_sampled_path, f'flow_sampled_%06d_%06d.png'%(i+1,i)))
#         save_image(flow_fwd_gt_rgb, os.path.join(flow_fwd_gt_path, f'flow_gt_%06d_%06d.png'%(i,i+1)  ))
#         torch.save(samples_flow_fwd,  os.path.join(flow_fwd_sampled_path, f'flow_sampled_%06d_%06d.pt'%(i+1,i)  ))
#         torch.save(flow_fwd_gt, os.path.join(flow_fwd_gt_path, f'flow_gt_%06d_%06d.pt'%(i,i+1) ))




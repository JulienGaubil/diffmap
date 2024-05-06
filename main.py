import argparse, os, sys, glob, datetime, importlib, csv
import hydra
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl

from hydra import compose, initialize
from pathlib import Path
from jaxtyping import Float
from torch import Tensor
from packaging import version
from omegaconf import OmegaConf, DictConfig
from functools import partial
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.util import instantiate_from_config, modify_conv_weights
from flow_vis_torch import flow_to_color
from ldm.modules.flowmap.visualization.depth import color_map_depth
from ldm.modules.flowmap.visualization.color import apply_color_map_to_image

# Enable arithmetic division in .yaml file with keyword "divide".
OmegaConf.register_new_resolver("divide", (lambda x, y: x//y), replace=True)


MULTINODE_HACKS = False

@rank_zero_only
def rank_zero_print(*args):
    print(*args)

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config,
                 lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        if not self.debug and trainer.global_rank == 0:
            rank_zero_print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time
                time.sleep(5)
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            rank_zero_print("Lightning config")
            rank_zero_print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
            check_idx += 1
        else:
            should_log = self.check_frequency(check_idx)
        if (should_log and  (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or split == "val":
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class ImageLoggerDiffmap(ImageLogger):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__(batch_frequency, max_images, clamp, increase_log_steps,
                 rescale, disabled, log_on_batch_idx, log_first_step,
                 log_images_kwargs, log_all_val)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split, modality):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            # TODO - properly handle modalities
            if modality not in ['optical_flow', 'depth_trgt', 'depth_ctxt']:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, modality):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale and modality not in ['optical_flow', 'depth_trgt', 'depth_ctxt']:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def intermediates_figure(
            self,
            images: Float[Tensor, "batch viz=4 channel height width"],
            modality: str
            ) -> Float[Tensor, "batch rgb=3 height_viz width"]:
        intermediate_samples = list()
        for i in range(images.size(0)):
            if modality == 'optical_flow':
                flow_vizs_sample = flow_to_color(images[i,:,:2,:,:]) / 255 # gt - noised gt - denoising output - denoised gt
            else:
                flow_vizs_sample = images[i]
                # Concatenate visualization for a given sample.
            flow_vizs_sample = torch.cat([flow_vizs_sample[j] for j in range(flow_vizs_sample.size(0))], dim=1)
            intermediate_samples.append(flow_vizs_sample)
        
        return torch.stack(intermediate_samples, dim=0)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
            check_idx += 1
        else:
            should_log = self.check_frequency(check_idx)
        if (should_log and  (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or split == "val":
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            #sample logged images
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            #logs every modality
            for modality in images.keys():
                split_m = split+f"_{modality}"
                images_m = images[modality]
                for k in images_m:
                    N = min(images_m[k].shape[0], self.max_images)
                    images_m[k] = images_m[k][:N].clone()
                    if isinstance(images_m[k], torch.Tensor):
                        images_m[k] = images_m[k].detach().cpu()
                        if self.clamp and modality not in ["depth_trgt", "depth_ctxt"]:
                            images_m[k] = torch.clamp(images_m[k], -1., 1.)
                    if modality == "optical_flow":
                        if k.startswith('intermediates_'):
                            images_m[k] = self.intermediates_figure(images_m[k][:,:,:,:,:], modality=modality)
                        else:
                            try:
                                images_m[k] = (flow_to_color(images_m[k][:,:2,:,:]) / 255)
                            except Exception:
                                print(images_m[k].min(), images_m[k].max())
                                assert False

                    elif modality in ["depth_trgt", "depth_ctxt"]:
                        # TODO do it properly, remove correspondence weights from here
                        if k == 'samples':
                            images_m[k] = color_map_depth(images_m[k].squeeze(1))
                        elif k == 'inputs':
                            images_m[k] = (images_m[k] + 1) /2
                        elif k == 'correspondence_weights':
                            images_m[k] = apply_color_map_to_image(images_m[k], "gray")
                    
                    elif k.startswith('intermediates_'):
                            images_m[k] = self.intermediates_figure(images_m[k][:,:,:,:,:], modality=modality)

                self.log_local(pl_module.logger.save_dir, split_m, images_m,
                            pl_module.global_step, pl_module.current_epoch, batch_idx, modality)

                logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                logger_log_images(pl_module, images_m, pl_module.global_step, split_m, modality)

            if is_train:
                pl_module.train()

class ImageLoggerAutoEncoder(ImageLogger):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__(batch_frequency, max_images, clamp, increase_log_steps,
                 rescale, disabled, log_on_batch_idx, log_first_step,
                 log_images_kwargs, log_all_val)


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
            check_idx += 1
        else:
            should_log = self.check_frequency(check_idx)
        if (should_log and  (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or split == "val":
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)
                if pl_module.image_key == "optical_flow":
                        images[k] = (flow_to_color(images[k][:,:2,:,:]) / 255)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class SingleImageLogger(Callback):
    """does not save as grid but as single images"""
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_always=False):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_always = log_always

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)
        for k in images:
            subroot = os.path.join(root, k)
            os.makedirs(subroot, exist_ok=True)
            base_count = len(glob.glob(os.path.join(subroot, "*.png")))
            for img in images[k]:
                if self.rescale:
                    img = (img + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
                img = img.transpose(0, 1).transpose(1, 2).squeeze(-1)
                img = img.numpy()
                img = (img * 255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}_{:08}.png".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx,
                    base_count)
                path = os.path.join(subroot, filename)
                Image.fromarray(img).save(path)
                base_count += 1

    def log_img(self, pl_module, batch, batch_idx, split="train", save_dir=None):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or self.log_always:
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir if save_dir is None else save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
                check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
            return True
        return False

@hydra.main(
        version_base=None,
        config_path='configs',
        config_name='common'
)
def run(config: DictConfig) -> None:
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
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Force load config references.
    config = OmegaConf.to_container(config, resolve=True)
    config = OmegaConf.create(config)

    if config.experiment_cfg.name and config.experiment_cfg.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if config.experiment_cfg.resume is not None:
        if not os.path.exists(config.experiment_cfg.resume):
            raise ValueError("Cannot find {}".format(config.experiment_cfg.resume))
        if os.path.isfile(config.experiment_cfg.resume):
            paths = config.experiment_cfg.resume.split("/")
            # idx = len(paths)-paths[::-1].index("logs")+1
            # logdir = "/".join(paths[:idx])
            logdir = "/".join(paths[:-2])
            ckpt = config.experiment_cfg.resume
        else:
            assert os.path.isdir(config.experiment_cfg.resume), config.experiment_cfg.resume
            logdir = config.experiment_cfg.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        config.experiment_cfg.resume_from_checkpoint = ckpt
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        config.experiment_cfg.resume = ""
        if config.experiment_cfg.name:
            name = "_" + config.experiment_cfg.name
        else:
            name = ""
        nowname = now + name + config.experiment_cfg.postfix
        logdir = os.path.join(config.experiment_cfg.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(config.experiment_cfg.seed)

    try:
        # default to ddp
        if config.lightning.trainer.gpus is None or config.lightning.trainer.gpus == []:
            del config.lightning.trainer.accelerator
            cpu = True
        else:
            rank_zero_print(f"Running on GPUs {config.lightning.trainer.gpus}")
            cpu = False
            if len(config.lightning.trainer.gpus) <= 1:
                del config.lightning.trainer.accelerator
        trainer_opt = argparse.Namespace(**config.lightning.trainer)

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
            out_filters_load = old_state["model.diffusion_model.out.2.weight"]
            out_filters_current = new_state["model.diff_out.2.weight"]
            out_shape = out_filters_current.shape

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

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": config.experiment_cfg.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in config.lightning:
            logger_cfg = config.lightning.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in config.lightning:
            modelckpt_cfg = config.lightning.modelcheckpoint
        else:
            modelckpt_cfg =  OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        rank_zero_print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse('1.4.0'):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": config.experiment_cfg.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": config.lightning,
                    "debug": config.experiment_cfg.debug,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    # "log_momentum": True
                }
            },
            "cuda_callback": {
                "target": "main.CUDACallback"
            },
        }
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

        if "callbacks" in config.lightning:
            callbacks_cfg = config.lightning.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
            rank_zero_print(
                'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
            default_metrics_over_trainsteps_ckpt_dict = {
                'metrics_over_trainsteps_checkpoint':
                    {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                    'params': {
                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        'save_top_k': -1,
                        'every_n_train_steps': 10000,
                        'save_weights_only': True
                    }
                    }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
            callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
        elif 'ignore_keys_callback' in callbacks_cfg:
            del callbacks_cfg['ignore_keys_callback']

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()
        if not config.lightning.get("find_unused_parameters", True):
            from pytorch_lightning.plugins import DDPPlugin
            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))
        if MULTINODE_HACKS:
            # disable resume from hpc ckpts
            # NOTE below only works in later versions
            # from pytorch_lightning.plugins.environments import SLURMEnvironment
            # trainer_kwargs["plugins"].append(SLURMEnvironment(auto_requeue=False))
            # hence we monkey patch things
            from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
            setattr(CheckpointConnector, "hpc_resume_path", None)

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
        trainer.logdir = logdir  ###

        #cree et prepare les dataloaders/datasets
        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup() #cree les datasets
        rank_zero_print("#### Data #####")
        try:
            for k in data.datasets:
                rank_zero_print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
        except:
            rank_zero_print("datasets not yet initialized.")

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(config.lightning.trainer.gpus)
        else:
            ngpu = 1
        if 'accumulate_grad_batches' in config.lightning.trainer:
            accumulate_grad_batches = config.lightning.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        config.lightning.trainer.accumulate_grad_batches = accumulate_grad_batches
        if config.experiment_cfg.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            rank_zero_print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            rank_zero_print("++++ NOT USING LR SCALING ++++")
            rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")


        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                rank_zero_print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)


        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb;
                pudb.set_trace()


        import signal

        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if config.experiment_cfg.train:
            try:
                trainer.fit(model, data)
            except Exception:
                if not config.experiment_cfg.debug:
                    melk()
                raise
        if not config.experiment_cfg.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except RuntimeError as err:
        if MULTINODE_HACKS:
            import requests
            import socket
            device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            hostname = socket.gethostname()
            ts = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            resp = requests.get('http://169.254.169.254/latest/meta-data/instance-id')
            rank_zero_print(f'ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}', flush=True)
        raise err
    except Exception:
        if config.experiment_cfg.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if config.experiment_cfg.debug and not config.experiment_cfg.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            rank_zero_print(trainer.profiler.summary())


if __name__ == "__main__":
    run()
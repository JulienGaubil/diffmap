import os, sys, glob, datetime, importlib, csv
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl

from pathlib import Path
from jaxtyping import Float
from torch import Tensor
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from flow_vis_torch import flow_to_color
from einops import rearrange

from ldm.misc.util import rank_zero_print
from ldm.visualization import color_map_depth, apply_color_map_to_image, filter_depth, warp_image_flow, overlay_images


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


class ImageLoggerDiffmap(ImageLogger):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None, log_all_val=False):
        super().__init__(batch_frequency, max_images, clamp, increase_log_steps,
                 rescale, disabled, log_on_batch_idx, log_first_step,
                 log_images_kwargs, log_all_val)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split, nrow):
        """Create grid visualization and log for tensorboard.
        """
        for k in images:
            images_viz = images[k]
            nrow_ = nrow if images_viz.size(0) // nrow > 1 else images_viz.size(0)
            grid = torchvision.utils.make_grid(images_viz, nrow=nrow_)
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx, nrow):
        """Prepare and save local logs.
        """
        root = os.path.join(save_dir, "images", split)

        # Log every visualization for the modality.
        for key in images:
            # Create samples grid.
            images_viz = images[key]
            nrow_ = nrow if images_viz.size(0) // nrow > 1 else images_viz.size(0)
            grid = torchvision.utils.make_grid(images_viz, nrow=nrow_)
            grid = rearrange(grid, 'c h w -> h w c').squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)

            # Save image file.
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                key,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def prepare_images(self, images: Float[Tensor, "batch frame channel=3 height width"]) -> Float[Tensor, "(batch frame) channel=3 height width"]:
        """Subsample and copy tensors to cpu.
        """
        N = min(images.shape[0], self.max_images)
        images = images[:N].clone()
        images = rearrange(images, 'b f ... -> (b f) ...')
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
        return images

    def make_intermediates_figure(
        self,
        images: Float[Tensor, "sample viz=4 channel=3 height width"],
        modality: str
    ) -> Float[Tensor, "sample rgb=3 height_viz width"]:
        """Prepare figure for intermediate diffusion step visualization before passing to grid.
        """
        intermediate_samples = list()

        # Create a stacked visualization for every sample.
        for i in range(images.size(0)):
            if modality == 'optical_flow':
                flow_vizs_sample = flow_to_color(images[i,:,:2,:,:]) / 255 # gt - noised gt - denoising output - denoised gt
            else:
                flow_vizs_sample = images[i]
                # Concatenate visualization for a given sample.
            flow_vizs_sample = torch.cat([flow_vizs_sample[j] for j in range(flow_vizs_sample.size(0))], dim=1)
            intermediate_samples.append(flow_vizs_sample)
        
        # Stacks individual visualizations in batch.
        intermediate_figure = torch.stack(intermediate_samples, dim=0)
        return intermediate_figure
    
    def make_warped_figure(
      self,
      fwd_flows: Float[Tensor, "sample vu=2 height width"],
      src_images: Float[Tensor, "sample 3 height width"],
      trgt_images: Float[Tensor, "sample 3 height width"]
    ) -> Float[Tensor, "sample 3 height width"]:
        """Warp source images with optical flow and overlay with trgt image.
        """
        warped_images_figs = list()

        for k in range(fwd_flows.size(0)):
            warped_image = warp_image_flow(
                src_image=src_images[k],
                flow=fwd_flows[k]
            )
            overlayed_image = overlay_images(
                im1=trgt_images[k],
                im2=warped_image,
                alpha_im1=0.5
            ) # (c h w), pixel range [0,1]
            warped_images_figs.append(overlayed_image)

        warped_images_figs = torch.stack(warped_images_figs, dim=0)

        return warped_images_figs

    def prepare_log_depth(
        self,
        batch: dict,
        log_dict: dict[str, Float[Tensor, "batch frame channel height width"]],
        modality: str
    ) -> dict[str, Float[Tensor, "(batch frame) 3 height width"]]:
        logs_depth = dict()

        # Make figures for depth samples and inputs visualization.
        for key, visualization in log_dict.items():
            # Subsample and copy tensors.
            if len(visualization.size()) == 5 and visualization.size(2) == 3:
                visualization = visualization.mean(2)
            images = self.prepare_images(visualization)
            
            # TODO - do it properly, remove correspondence weights from here
            # Apply colorization.
            if key == 'correspondence_weights':
                images = apply_color_map_to_image(images, "gray")
            else:
                images, _ = filter_depth(images)
                images = color_map_depth(images)

            logs_depth[key] = images
        
        return logs_depth

    def prepare_log_flow(
        self,
        log_dict: dict[str, Float[Tensor, "batch frame channel height width"]],
        src_images,
        trgt_images,
        modality: str
    ) -> dict[str, Float[Tensor, "(batch frame) 3 height width"]]:
        logs_flow = dict()
        
        # Make figures for flow samples and inputs visualization.
        for key, visualization in log_dict.items():
            # Subsample and copy tensors.
            if key.startswith('intermediates_'):
                b, v, f, v, h, w = visualization.shape
                visualization = rearrange(visualization, 'b v f ... -> b f v ...')
            images = self.prepare_images(visualization)

            # if key == "samples":
            #     fwd_flows = images[:,:2,:,:]
            #     src_images = (self.prepare_images(src_images) + 1.0) / 2.0
            #     trgt_images = (self.prepare_images(trgt_images) + 1.0) / 2.0
            #     warped_images = self.make_warped_figure(
            #         src_images=src_images,
            #         trgt_images=trgt_images,
            #         fwd_flows=fwd_flows
            #     )
            #     logs_flow["warped_fwd"] = warped_images

            # Apply colorization.
            if key.startswith('intermediates_'):
                images = rearrange(images, '(b f) v c h w -> b f v c h w', f=f)
                images = images[:,0,...]
                images = self.make_intermediates_figure(images, modality=modality)
            else:
                images = flow_to_color(images[:,:2,:,:]) / 255
            
            logs_flow[key] = images

        return logs_flow

    def prepare_log_rgb(
        self,
        batch: dict,
        log_dict: dict[str, Float[Tensor, "batch frame 3 height width"]],
        modality: str
    ) -> dict[str, Float[Tensor, "* 3 height width"]]:
        logs_rgb = dict()

        # Create figures for RGB samples and inputs visualization.
        for key, visualization in log_dict.items():
            # Prepare image tensors from [-1,1] to [0,1].
            if key.startswith('intermediates_'):
                b, v, f, v, h, w = visualization.shape
                visualization = rearrange(visualization, 'b v f ... -> b f v ...')
            images = self.prepare_images(visualization)

            # Make RGB figures.
            if self.clamp:
                images = torch.clamp(images, -1., 1.)
            images = (images + 1.0) / 2.0 #  [-1,1] -> [0,1].
            if key.startswith('intermediates_'):
                images = rearrange(images, '(b f) v c h w -> b f v c h w', f=f)
                images = images[:,0,...]
                images = self.make_intermediates_figure(images, modality=modality)

            logs_rgb[key] = images

        return logs_rgb

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """Sample the diffusion model with a batch and log the results.
        """
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
            check_idx += 1
        else:
            should_log = self.check_frequency(check_idx)

        # Sample model with batch and log results.
        if (should_log and (check_idx % self.batch_freq == 0) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0) or split == "val":
            logger = type(pl_module.logger)

            # Sample model.
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                log_dict = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

            # Prepare visualization for every modality.
            logs = dict()
            for modality, log_dict_modality in log_dict.items():
                if "depth" in modality:
                    logs[modality] = self.prepare_log_depth(batch, log_dict_modality, modality)
                elif modality == "optical_flow":
                    logs[modality] = self.prepare_log_flow(
                        log_dict=log_dict_modality,
                        src_images=rearrange(batch[pl_module.cond_stage_key], 'b f h w c -> b f c h w'),
                        trgt_images=rearrange(batch[pl_module.first_stage_key], 'b f h w c -> b f c h w'),
                        modality=modality
                    )
                else:
                    logs[modality] = self.prepare_log_rgb(batch, log_dict_modality, modality)

            # Log every modality.
            for modality, log_dict_modality in logs.items():
                if pl_module.n_future > 1:
                    nrow = pl_module.n_future 
                else:
                    nrow = min(self.log_images_kwargs['N'], self.max_images)

                split_modality = split+f"_{modality}"

                # Locally save visualizations in log folder.
                self.log_local(pl_module.logger.save_dir, split_modality, log_dict_modality,
                            pl_module.global_step, pl_module.current_epoch, batch_idx, nrow)
                # Save visualizations for tensorboard.
                logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
                logger_log_images(pl_module, log_dict_modality, pl_module.global_step, split_modality, nrow)
            
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
import os, glob
import numpy as np
import torch
import torchvision
import pytorch_lightning as pl

from contextlib import nullcontext
from jaxtyping import Float
from torch import Tensor
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from flow_vis_torch import flow_to_color
from einops import rearrange, repeat
from omegaconf import OmegaConf
from dataclasses import fields

from ldm.misc.util import rank_zero_print
from ldm.misc.modalities import Modality
from ldm.visualization.utils import prepare_visualization
from ldm.modules.flowmap.model.model_wrapper_pretrain import ModelOutput
from ldm.modules.flowmap.dataset.types import Batch
from ldm.modules.flowmap.flow import Flows
from ldm.modules.flowmap.visualization import VisualizerSummaryCfg, VisualizerSummary, VisualizerTrajectoryCfg, VisualizerTrajectory
from ldm.modules.flowmap.config.common import get_typed_root_config_diffmap


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
    def __init__(
        self, batch_frequency, max_images, clamp=True, increase_log_steps=True,
        rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
        log_images_kwargs=None, log_all_val=False
    ):
        super().__init__(
            batch_frequency, max_images, clamp, increase_log_steps,
            rescale, disabled, log_on_batch_idx, log_first_step,
            log_images_kwargs, log_all_val
        )

        # Instantiate trajectory visualizer.
        config_traj = {'name': "trajectory"}
        config_traj = OmegaConf.create(config_traj)
        cfg_traj = get_typed_root_config_diffmap(config_traj, VisualizerTrajectoryCfg)
        self.viz_trajectory = VisualizerTrajectory(cfg_traj)

        # Instantiate flowmap summary visualizer.
        config_viz = {'name': "summary", 'num_vis_frames': 1000}
        config_viz = OmegaConf.create(config_viz)
        cfg_viz = get_typed_root_config_diffmap(config_viz, VisualizerSummaryCfg)
        self.viz_summary = VisualizerSummary(cfg_viz)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split, nrow):
        """Create grid visualization and log for tensorboard.
        """
        for k in images:
            images_viz = images[k]
            nrow_ = nrow if nrow > 1 else images_viz.size(0)
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
            nrow_ = nrow if nrow > 1 else images_viz.size(0)
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

    def prepare_images(self, images: Float[Tensor, "sample frame channel=3 height width"]) -> Float[Tensor, "(sample frame) channel=3 height width"]:
        """Subsample and copy tensors to cpu.
        """
        N = min(images.shape[0], self.max_images)
        images = images[:N].clone()
        images = rearrange(images, 'b f ... -> (b f) ...')
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu()
        return images
    
    # def make_warped_figure(
    #   self,
    #   fwd_flows: Float[Tensor, "sample vu=2 height width"],
    #   src_images: Float[Tensor, "sample 3 height width"],
    #   trgt_images: Float[Tensor, "sample 3 height width"]
    # ) -> Float[Tensor, "sample 3 height width"]:
    #     """Warp source images with optical flow and overlay with trgt image.
    #     """
    #     warped_images_figs = list()

    #     for k in range(fwd_flows.size(0)):
    #         warped_image = warp_image_flow(
    #             src_image=src_images[k],
    #             flow=fwd_flows[k]
    #         )
    #         overlayed_image = overlay_images(
    #             im1=trgt_images[k],
    #             im2=warped_image,
    #             alpha_im1=0.5
    #         ) # (c h w), pixel range [0,1]
    #         warped_images_figs.append(overlayed_image)

    #     warped_images_figs = torch.stack(warped_images_figs, dim=0)

    #     return warped_images_figs
    

    def log_visualization(
        self,
        pl_module: pl.LightningModule,
        visualization: Float[Tensor, "(sample frame) 3 height width"],
        modality: Modality,
        logger,
        split: str,
        batch_idx: int,
        label_visualization: str,
        nrow: int | None = None
    ) -> None:
        if nrow is None:
            if pl_module.n_future > 1:
                nrow = pl_module.n_future
            else:
                nrow = min(self.log_images_kwargs['N'], self.max_images)
        
        label_log_folder = '/'.join(['_'.join([split, modality.modality]), modality.name])
        log_dict_modality = {label_visualization: visualization}

        # Save visualizations locally.
        self.log_local(
            pl_module.logger.save_dir,
            label_log_folder,
            log_dict_modality,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            nrow
        )
        # Save visualizations with tensorboard.
        logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
        logger_log_images(pl_module, log_dict_modality, pl_module.global_step, label_log_folder, nrow)

    @torch.no_grad()
    def log_intermediate_samples(
        self,
        pl_module: pl.LightningModule,
        x: Float[Tensor, "sample noisy_channel height width"],
        c: Float[Tensor, "sample cond_channel height width"],
        logger,
        split: str,
        batch_idx: int
    ) -> None:
        """Sample and log intermediate denoising step visualization.
        """
        # Sample intermediate diffusion step.
        t_intermediate = pl_module.num_timesteps // 2
        x_intermediate, samples_intermediate = pl_module.sample_intermediate(x_start=x, cond=c, t=t_intermediate)

        # Prepare every intermediate tensor.
        visualization_all = {modality._id: list() for modality in pl_module.modalities_out}
        for intermediate_tensor in [x, x_intermediate, samples_intermediate.x_denoised, samples_intermediate.x_recon]: # visualized tensors
            intermediate_samples = rearrange(intermediate_tensor, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
            intermediate_samples = pl_module.modalities_out.split_modalities_multiplicity(intermediate_samples, modality_ids=pl_module.modalities_out.ids_denoised)
            
            # Prepare intermediate visualization for every modality.
            for modality in pl_module.modalities_out.denoised_modalities:
                B = intermediate_samples[modality._id].size(0)
                visualization_intermediate = prepare_visualization(intermediate_samples[modality._id], modality, sample=True)
                visualization_intermediate = rearrange(visualization_intermediate, '(b f) c h w -> b f c h w', b=B)
                visualization_all[modality._id].append(visualization_intermediate[:,0]) # (sample, channel, height, width) subsample to select only first future frame intermediate visualization

        # Log all intermediate visualizations for every denoised modality.
        for modality in pl_module.modalities_out.denoised_modalities:
            visualization = torch.stack(visualization_all[modality._id], dim=0) # (v b c h w)
            visualization = rearrange(visualization, 'v b c h w -> (v b) c h w')
            self.log_visualization(
                pl_module,
                visualization,
                modality,
                logger,
                split,
                batch_idx,
                f'intermediates_{t_intermediate}',
                nrow=B
            )

        # Log intermediates for clean modalities.
        weights = samples_intermediate.weights
        samples_prediction = rearrange(samples_intermediate.x_predicted, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
        samples_prediction = pl_module.modalities_out.split_modalities_multiplicity(samples_prediction, modality_ids=pl_module.modalities_out.ids_clean)
        for modality in pl_module.modalities_out.clean_modalities:
            sample_m = samples_prediction[modality._id]
            visualization = prepare_visualization(sample_m, modality, sample=True)
            self.log_visualization(
                pl_module,
                visualization,
                modality,
                logger,
                split,
                batch_idx,
                f'intermediates_{t_intermediate}',
            )

        # Log intermediates for correspondence weight samples.
        if weights is not None:
            modality_weight = Modality(name='correspondence', modality='weight', multiplicity=weights.size(1), channels_m=0, denoised=False)
            visualization = prepare_visualization(weights, modality_weight)
            self.log_visualization(
                pl_module,
                visualization,
                modality_weight,
                logger,
                split,
                batch_idx,
                f'intermediate_sample_{t_intermediate}',
            )

    def log_intermediate_inputs(
        self,
        pl_module: pl.LightningModule,
        x: Float[Tensor, "sample noisy_channel height width"],
        logger,
        split: str,
        batch_idx: int
    ) -> None:
        """Log forward diffusion process intermediate steps.
        """
        B = x.size(0)
        visualization_all = {modality._id: list() for modality in pl_module.modalities_in.denoised_modalities}

        # Get diffusion intermediates.
        for t in range(pl_module.num_timesteps):
            if t % pl_module.log_every_t == 0 or t == pl_module.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=B)
                t = t.to(pl_module.device).long()
                noise = torch.randn_like(x)

                x_noisy = pl_module.q_sample(x_start=x, t=t, noise=noise) # forward process
                x_noisy = rearrange(x_noisy, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
                x_noisy_split = pl_module.modalities_in.split_modalities_multiplicity(x_noisy, modality_ids=pl_module.modalities_in.ids_denoised)

                for modality in pl_module.modalities_in.denoised_modalities:
                    visualization_intermediate = prepare_visualization(x_noisy_split[modality._id], modality)
                    visualization_intermediate = rearrange(visualization_intermediate, '(b f) c h w -> b f c h w', b=B)
                    visualization_all[modality._id].append(visualization_intermediate[:,0]) # (sample, channel, height, width) subsample to select only first future frame intermediate visualization

        # Log all intermediate visualizations for every noisy modality.
        for modality in pl_module.modalities_in.denoised_modalities:
            n = len(visualization_all[modality._id])
            diffusion_grid = torch.stack(visualization_all[modality._id], dim=1) # (b n c h w)
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            self.log_visualization(
                pl_module,
                diffusion_grid,
                modality,
                logger,
                split,
                batch_idx,
                'forward_diffusion_row',
                nrow=n
            )

    def log_flowmap(
        self,
        pl_module: pl.LightningModule,
        flowmap_output: ModelOutput,
        flows: Flows,
        flowmap_batch: dict, 
        logger,
        split: str,
        batch_idx: int
    ) -> None:
        
        # Prepare visualization inputs.
        B, F, H, W = flowmap_output.depths.shape
        idx_viz = np.random.randint(B)
        batch_viz = Batch(**flowmap_batch)[idx_viz:idx_viz+1]
        flows_viz = flows[idx_viz:idx_viz+1]
        model_output_viz = ModelOutput(**{field.name: getattr(flowmap_output, field.name)[idx_viz:idx_viz+1] for field in fields(flowmap_output)})
        model_viz = None
        global_step_viz = 1
        
        # Visualize depth and pose-induced optical flow.
        summary_viz = self.viz_summary.visualize(
            batch_viz,
            flows_viz,
            None,
            model_output_viz,
            model_viz,
            global_step_viz
        )
        summary_fig = summary_viz['summary'][None].detach().cpu()

        # Prepare trajectory visualization.
        batch_viz.extrinsics = model_output_viz.extrinsics
        summary_traj = self.viz_trajectory.visualize(
            batch_viz,
            flows_viz,
            None,
            model_output_viz,
            model_viz,
            global_step_viz
        )
        summary_traj_viz = summary_traj['trajectory'][None].detach().cpu()


        # Logs visualizations.
        label_log_folder = '_'.join([split,'flowmap'])
        log_dict_modality = {
            'summary': summary_fig,
            'trajectory': summary_traj_viz
        }
        nrow = 1
        self.log_local(
            pl_module.logger.save_dir,
            label_log_folder,
            log_dict_modality,
            pl_module.global_step,
            pl_module.current_epoch,
            batch_idx,
            nrow
        )
        logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
        logger_log_images(pl_module, log_dict_modality, pl_module.global_step, label_log_folder, nrow)

        depths = flowmap_output.depths.detach().cpu() # (B, F, H, W), depth maps
        surfaces = flowmap_output.surfaces.detach().cpu() # (B, F, H, W, XYZ=3), local pointmaps
        intrinsics = flowmap_output.intrinsics.detach().cpu() # (B, F, 3, 3), estimated camera intrinsics
        extrinsics = flowmap_output.extrinsics.detach().cpu() # (B, F, 4, 4), estimated camera extrinsics
        weights = flowmap_output.backward_correspondence_weights.detach().cpu() # (B, F, H, W), backward correspondence weights

        # Log points clouds.
        vertices_tensor = rearrange(surfaces[idx_viz], 'f h w xyz -> f (h w) xyz')
        color_tensor = flowmap_batch['videos'][idx_viz].detach().cpu()
        colors_tensor = (rearrange(color_tensor, 'f c h w -> f (h w) c') * 255).to(torch.int8)
        config_dict = {
            'point_size': 10  # Increase point size
        }
        pl_module.logger.experiment.add_mesh(f'{split}/flowmap_surfaces', vertices=vertices_tensor, colors=colors_tensor, config_dict= config_dict, global_step=pl_module.global_step)
        

    def log_inputs(
        self,
        pl_module: pl.LightningModule,
        batch: dict,
        logger,
        split: str,
        batch_idx: int,
        N: int = 8,
        plot_diffusion_rows: bool = False,
        **kwargs
    ) -> None:
        """Log model inputs.
        """
        # Get model inputs
        x, c, xc = pl_module.get_input(
            batch,
            force_c_encode=True,
            return_original_cond=True,
            bs=N
        )

        # Split input and conditioning modalities.
        x_log = rearrange(x, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
        xc_log = rearrange(xc, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
        x_split = pl_module.modalities_in.split_modalities_multiplicity(x_log)
        xc_split = pl_module.modalities_cond.split_modalities_multiplicity(xc_log)

        # Prepare and log conditioning visualizations.
        for modality in pl_module.modalities_cond:
            if not modality in pl_module.modalities_in:
                x_m = xc_split[modality._id]
                visualization = prepare_visualization(x_m, modality)
                self.log_visualization(
                    pl_module,
                    visualization,
                    modality,
                    logger,
                    split,
                    batch_idx,
                    'conditioning',
                    nrow=pl_module.n_ctxt
                )

        # Log input visualizations.
        for modality in pl_module.modalities_in:
            x_m = x_split[modality._id]
            visualization = prepare_visualization(x_m, modality)
            self.log_visualization(
                pl_module,
                visualization,
                modality,
                logger,
                split,
                batch_idx,
                'input'
            )

        if plot_diffusion_rows:
            self.log_intermediate_inputs(pl_module, x, logger, split, batch_idx)
    
    @torch.no_grad()
    def log_samples(
        self,
        pl_module: pl.LightningModule,
        batch: dict,
        logger,
        split: str,
        batch_idx: int,
        sample: bool = True,
        N: int = 8,
        n_row: int = 4,
        use_ema_scope: bool = True,
        use_ddim: bool = True,
        ddim_steps: int = 200,
        ddim_eta: float = 1.,
        plot_denoise_rows: bool = False,
        plot_progressive_rows: bool = False,
        unconditional_guidance_scale: float = 1.,
        unconditional_guidance_label = None,
        **kwargs
    ) -> None:
        """Sample model and log samples.
        """
        # Get model inputs.
        x, c, flows, xc = pl_module.get_input(
            batch,
            force_c_encode=True,
            return_original_cond=True,
            bs=N,
            return_flows=True
        )

        ema_scope = pl_module.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None and pl_module.num_timesteps > 1

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)

        # Sample diffusion model and log samples.
        if sample:
            # Sampling.
            with ema_scope("Sampling"):
                samples, x_denoise_row = pl_module.sample_log(
                    cond=c,
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta
                )

            if pl_module.use_flowmap_loss:
                dummy_flowmap_batch, flows, depths, correspondence_weights, surfaces, intrinsics = pl_module.get_input_flowmap(samples.x_predicted, flows, samples.weights)
                _, flowmap_output, metrics_dict = pl_module.flowmap_loss_wrapper(
                    dummy_flowmap_batch,
                    flows,
                    depths,
                    correspondence_weights,
                    pl_module.global_step,
                    surfaces,
                    intrinsics,
                    return_outputs=True
                )

            weights = samples.weights
            samples_diffusion = rearrange(samples.x_denoised, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
            samples_prediction = rearrange(samples.x_predicted, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
            samples_diffusion = pl_module.modalities_out.split_modalities_multiplicity(samples_diffusion, modality_ids=pl_module.modalities_out.ids_denoised)
            samples_prediction = pl_module.modalities_out.split_modalities_multiplicity(samples_prediction, modality_ids=pl_module.modalities_out.ids_clean)
            samples = dict(samples_diffusion, **samples_prediction)

            # Log samples and gt.
            for modality in pl_module.modalities_out:
                sample_m = samples[modality._id]
                visualization = prepare_visualization(sample_m, modality, sample=True, pointmap_mapping_func=pl_module.geometric_modalities.pointmap_mapping_func) # TODO - remove hack
                self.log_visualization(
                    pl_module,
                    visualization,
                    modality,
                    logger,
                    split,
                    batch_idx,
                    'sample'
                )

                if modality._id in batch.keys() and modality not in pl_module.modalities_in: #gt
                    x_m = pl_module.get_input_modality(batch, modality._id)[:N]
                    x_m = rearrange(x_m, 'b (f c) h w -> b f c h w', c=modality.channels_m)
                    visualization = prepare_visualization(x_m, modality)
                    self.log_visualization(
                        pl_module,
                        visualization,
                        modality,
                        logger,
                        split,
                        batch_idx,
                        'gt'
                    )

            # Log correspondence weight samples.
            if weights is not None:
                modality_weight = Modality(name='correspondence', modality='weight', multiplicity=weights.size(1), channels_m=0, denoised=False)
                visualization = prepare_visualization(weights, modality_weight)
                self.log_visualization(
                    pl_module,
                    visualization,
                    modality_weight,
                    logger,
                    split,
                    batch_idx,
                    'sample'
                )

            # Visualize intermediate diffusion step.
            if pl_module.modalities_out.n_noisy_channels > 0:
                self.log_intermediate_samples(pl_module, x, c, logger, split, batch_idx)

            if pl_module.use_flowmap_loss:
                self.log_flowmap(pl_module, flowmap_output, flows, dummy_flowmap_batch, logger, split, batch_idx)
                
            
            # Visualize intermediate diffusion rows.
            if plot_denoise_rows and pl_module.channels > 0:
                raise NotImplementedError
                # for modality in pl_module.modalitie_out:
                #     denoise_grid = pl_module._get_denoise_row_from_list(x_denoise_row, modality=modality._id) # TODO - adapt to every modality
                #     log[modality._id]["denoise_row"] = denoise_grid

        # Sampling with classifier-free guidance.
        if unconditional_guidance_scale > 1.0 and pl_module.model.conditioning_key not in ["concat", "hybrid"]:
            raise NotImplementedError
            # uc = pl_module.get_unconditional_conditioning(N, unconditional_guidance_label)
            # # uc = torch.zeros_like(c)
            # with ema_scope("Sampling with classifier-free guidance"):
            #     samples_cfg, _ = pl_module.sample_log(cond=c, batch_size=N, ddim=use_ddim,
            #                                     ddim_steps=ddim_steps, eta=ddim_eta,
            #                                     unconditional_guidance_scale=unconditional_guidance_scale,
            #                                     unconditional_conditioning=uc,
            #                     )
            #     samples_cfg = rearrange(samples_cfg.x_noisy, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
            #     samples_cfg = pl_module.modalities_out.split_modalities_multiplicity(samples_cfg, modality_ids=pl_module.modalities_out.ids_denoised)

            # # Log classifier-free guidance samples.
            # x_samples_cfg = samples_cfg[id_m]
            # log[id_m][f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        # Plotting progressive denoising row.
        if plot_progressive_rows:
            raise NotImplementedError
            # with ema_scope("Plotting Progressives"):
            #     _, progressives = pl_module.progressive_denoising(c,
            #                                                 shape=(pl_module.channels, pl_module.image_size, pl_module.image_size),
            #                                                 batch_size=N)
                
            #     for k in range(len(progressives)):
            #         s = rearrange(s, 'b (f c) h w -> b f c h w', c=pl_module.channels_m)
            #         progressives[k] = pl_module.modalities_out.split_modalities_multiplicity(s, modality_ids=pl_module.modalities_out.ids_denoised)

            # progressives_m = [s[id_m] for s in progressives]
            # prog_row = pl_module._get_denoise_row_from_list(progressives_m, desc="Progressive Generation", modality=id_m)
            # log[id_m][f"progressive_row"] = prog_row

    @torch.no_grad()
    def log_img(
        self,
        pl_module: pl.LightningModule,
        batch: dict,
        batch_idx: int,
        split: str = "train",
    ) -> dict:
        """Sample diffusion model and create visualizations for logging.
        """
        # Check if log should be performed.
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
            check_idx += 1
        else:
            should_log = self.check_frequency(check_idx)

        # Sample model with batch and log results.
        if (should_log and (check_idx % self.batch_freq == 0) and hasattr(pl_module, "log_images") and
            callable(pl_module.log_images) and self.max_images > 0) or split == "val":
            
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            # Log model inputs and samples.
            self.log_inputs(pl_module, batch, logger, split, batch_idx, **self.log_images_kwargs)
            self.log_samples(pl_module, batch, logger, split, batch_idx, **self.log_images_kwargs)

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
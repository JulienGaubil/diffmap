# import torch
# import torch.nn as nn
# import numpy as np
# import pytorch_lightning as pl
# from torch.optim.lr_scheduler import LambdaLR
# from einops import rearrange, repeat
# from contextlib import contextmanager, nullcontext
# from functools import partial
# import itertools
# from tqdm import tqdm
# from torchvision.utils import make_grid
# from pytorch_lightning.utilities.distributed import rank_zero_only
# from omegaconf import ListConfig

# from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
# from ldm.modules.ema import LitEma
# from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
# from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
# from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
# from ldm.models.diffusion.ddim import DDIMSampler
# from ldm.modules.attention import CrossAttention

# from jaxtyping import Float
# from torch import Tensor
# from ldm.modules.flowmap.model.model_wrapper_pretrain import FlowmapLossWrapper
# from ldm.modules.flowmap.config.common import get_typed_root_config_diffmap
# from ldm.modules.flowmap.config.pretrain import DiffmapCfg
# from ldm.modules.flowmap.loss import get_losses
# from ldm.modules.flowmap.model.model import FlowmapModelDiff


# __conditioning_keys__ = {'concat': 'c_concat',
#                          'crossattn': 'c_crossattn',
#                          'adm': 'y'}

from .ddpm import *

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}






class FlowMapDiffusion(LatentDiffusion): #derived from LatentInpaintDiffusion
    """
    can either run as pure inpainting model (only concat mode) or with mixed conditionings,
    e.g. mask as concat and text via cross-attn.
    To disable finetuning mode, set finetune_keys to None

    Simultaneous diffusion of flow, depths and next frame
    """
    def __init__(self,  *args, modalities=['nfp'], first_stage_flow_config=None, flowmap_loss_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.modalities = modalities

        if first_stage_flow_config is not None:
            model = instantiate_from_config(first_stage_flow_config)
            self.first_stage_model_flow = model.eval()  #encodeur-decoder VAE
            self.first_stage_model_flow.train = disabled_train
            for param in self.first_stage_model_flow.parameters():
                param.requires_grad = False

        assert self.channels % len(self.modalities) == 0, "Number of channels should be a multiple of number of modalities"
        self.channels_m = self.channels // len(self.modalities) #number of individual channels per modality

        if flowmap_loss_config is not None:
            self.flowmap_loss_wrapper = self.init_flowmap_loss(flowmap_loss_config)
            
    def init_flowmap_loss(self, cfg_dict):
        cfg = get_typed_root_config_diffmap(cfg_dict, DiffmapCfg)
        # Set up the model.
        model = FlowmapModelDiff(cfg.model)
        losses = get_losses(cfg.loss)
        flowmap_loss_wrapper = FlowmapLossWrapper(
            cfg.model_wrapper,
            cfg.cropping,
            model,
            losses,
        )
        return flowmap_loss_wrapper
        
    def split_modalities(
            self,
            z: Float[Tensor, "_ 4*C _ _"],
            C: int = None,
            modalities: list[str] | None = None
        ) -> dict[str, Float[Tensor, "_ C _ _"]]:
        # Splits input tensor along every modality of chunk size C
        C = default(C, self.channels_m)
        modalities = default(modalities, self.modalities)
        split_all = dict(zip(self.modalities, torch.split(z, C, dim=1)))
        out = {m: split_all[m] for m in modalities}
        return out
    
    def decode_first_stage_modality(
            self,
            z_m: Float[Tensor, "_ C _ _"],
            modality: str,
            # compute_grxads: bool = False,
    ) -> Float[Tensor, "_ 3 _ _"]:
        # Decodes individual modality
        use_grad = torch.is_grad_enabled()
        with torch.set_grad_enabled(use_grad):
            first_stage_model = self.first_stage_model if modality != "optical_flow" else self.first_stage_model_flow
            z_m = 1. / self.scale_factor * z_m
            x_sample_m = first_stage_model.decode(z_m)
            return x_sample_m
    
    def decode_first_stage_all(
            self,
            z: Float[Tensor, "_ C _ _"],
            modalities: list[str] | None = None
        ) -> dict[str, Float[Tensor, "_ 3 _ _"]]:
        # Decodes input modalities
        modalities = default(modalities, self.modalities)
        z_split = self.split_modalities(z, modalities=modalities)

        x_sample = dict()
        for modality in modalities:
            z_m = z_split[modality]
            x_sample[modality] = self.decode_first_stage_modality(z_m, modality)
        return x_sample

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                    cond_key=None, return_original_cond=False, bs=None, return_x=False, return_flows_depths=False):
        '''
        Returns the inputs and outputs required for a diffusion sampling process and to supervise the model.
        Inputs:
            - k: string, self.first_stage_key, key for target image x_0 in the batch
        Output:
            - (default): list [z, c] with z embedding in VAE latent space of target image x_0, c conditioning signal
            - if return_first_stage_outputs: adds x_rec, decoding of z for reconstruction of input image by VAE (without diffusion sampling)
            - if force_c_encode: c is the feature encoding of the conditioning signal (eg with CLIP)
            - if return_x: adds x, target image x_0
            - if return_original_cond: adds xc, conditioning signal (non-encoded)
        '''
        x_list, z_list = list(), list()
        for modality in self.modalities:
            #encodes target image in VAE latent space
            x = DDPM.get_input(self, batch, modality) #image target clean x_0
            if bs is not None:
                x = x[:bs]
            x = x.to(self.device)
            if modality != "optical_flow":
                encoder_posterior = self.encode_first_stage(x)  #encode image target clean, latent clean encodé E(x_0)
                z = self.get_first_stage_encoding(encoder_posterior).detach() #sample espace latent VAE (dans ce cas, juste encoder_posterior scalé)
            else:
                # TODO faire proprement
                encoder_posterior = self.first_stage_model_flow.encode(x)
                z = self.scale_factor * encoder_posterior.sample()

            x_list.append(x)
            z_list.append(z)
        
        z = torch.cat(z_list, dim=1)
        x = torch.cat(x_list, dim=1)

        #gets conditioning image xc and encodes it with feature encoder eg CLIP
        if self.model.conditioning_key is not None:
            #gets conditioning image
            if cond_key is None: #mostly the case
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else: #mostly the case, cond_key is self.cond_stage_key and different from input imahe
                    xc = DDPM.get_input(self, batch, cond_key).to(self.device) #conditioning image
            else:
                xc = x

            #encodes conditioning image with feature encoder (eg CLIP)
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device)) #encoded conditioning
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else: #no conditioning in this case
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        
        #outputs
        out = [z, c] #z VAE latent for target image x_0, c encoding for conditioning signal
        if return_first_stage_outputs:
            x_rec_list = list()
            for k in range(len(self.modalities)):
                if self.modalities[k] != "optical_flow":
                    x_rec_list.append(self.decode_first_stage(z[:, k*4:(k+1)*4, ...]))
                else:
                    x_rec_list.append(self.first_stage_model_flow.decode((1/self.scale_factor) * z[:, k*4:(k+1)*4, ...])) # TODO faire proprement

            xrec = torch.cat(x_rec_list, dim=1)
            out.extend([x, xrec])
        if return_x:
            out.append(x)
        if return_flows_depths: #TODO properly handle modalities
            flows = {"forward": batch['optical_flow'].to(self.device), "backward": batch['optical_flow_bwd'].to(self.device)}
            flows_masks = {"forward": batch['optical_flow_mask'].to(self.device), "backward": batch['optical_flow_bwd_mask'].to(self.device)}
            correspondence_weights = batch['correspondence_weights'].to(self.device)
            if bs is not None:
                flows["forward"] = flows["forward"][:bs,:,:,:]
                flows_masks = flows["backward"][:bs,:,:]  
                correspondence_weights = correspondence_weights[:bs,:,:]             
            out.extend([flows, flows_masks, correspondence_weights])
        if return_original_cond:
            out.append(xc)
        return out

    def get_input_flowmap(
            self,
            x_recon_flowmap: Float[Tensor, "batch channels_latent height_latent width_latent"],
            flows: dict[str, Float[Tensor, "batch channels height width"]],
            flows_masks: dict[str, Float[Tensor, "batch height width"]],
            correspondence_weights: Float[Tensor, "batch height width"]
        ) -> tuple[dict[str, Tensor], dict[str, Tensor], Float[Tensor, "batch frame height width"]]:
        # Prepare depth, should be (batch frame height width)
        correspondence_weights = correspondence_weights[:, None, :, :]
        depths_recon = torch.stack([
            x_recon_flowmap["depth_ctxt"].mean(1),
            x_recon_flowmap["depth_trgt"].mean(1)
            ],
            dim=1
        )
        # Normalize the depth
        near = depths_recon.min()
        far = depths_recon.max()
        depths_recon = (depths_recon - near) / (far - near)
        depths_recon = depths_recon.clip(min=0, max=1)

        # Prepare flow
        # flows_recon = rearrange(x_recon_flowmap["optical_flow"][:, None, :2, :, :], 'b f xy w h -> b f w h xy') #estimated clean forward flow, TODO, should be (batch pair height width 2)
        flows_fwd = flows["forward"][:, None, :, :, :2]  #gt clean forward flows, TODO, should be (batch pair height width 2)
        flows_bwd = flows["backward"][:, None, :, :, :2]  #gt clean backward flows, TODO, should be (batch pair height width 2)
        flows_mask_fwd = flows_masks["forward"][:, None, :, :] #gt clean forward flows consistency masks, TODO, should be (batch pair height width)
        flows_mask_bwd = flows_masks["backward"][:, None, :, :] #gt clean backward flows consistency masks, TODO, should be (batch pair height width)
        
        flows = {
            "forward": flows_fwd,
            "backward": flows_bwd,
            "forward_mask": flows_mask_fwd,
            "backward_mask": flows_mask_bwd,
        }

        # Prepare flowmap dummy batch TODO remove hack
        N, _ ,_, H, W = flows_fwd.size()
        dummy_flowmap_batch = {
            "videos": torch.zeros((N, 2, 3, H, W), dtype=torch.float32, device=self.device),
            "indices": torch.tensor([0,1], device=self.device).repeat(N, 2),
            "scenes": [""],
            "datasets": [""],
        }

        return dummy_flowmap_batch, flows, depths_recon, correspondence_weights
    
    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False, modality=None):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            if modality != "optical_flow":
                denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
            else:
                denoise_row.append(self.first_stage_model_flow.decode((1/self.scale_factor) * zd.to(self.device))) #TODO faire proprement

        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
            

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        '''
        Performs diffusion forward and backward process on a given batch and returns logs for tensorboard
        '''
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        
        #gets conditioning image
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2]//25)
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2]//25)
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if sample: #sampling of the conditional diffusion model with DDIM for accelerated inference without logging intermediates
                # get denoise row
                with ema_scope("Sampling"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                            ddim_steps=ddim_steps,eta=ddim_eta) #samples generative process in latent space
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)

        #sampling with classifier free guidance
        if unconditional_guidance_scale > 1.0:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # uc = torch.zeros_like(c)
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                )

        if inpaint:
            # make a simple center square
            b, h, w = z.shape[0], z.shape[2], z.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):

                samples_inpaint, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                
            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples_outpaint, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)


        for k in range(len(self.modalities)):
            modality = self.modalities[k]
            log[modality] = dict()
            log[modality]["inputs"] = x[:, k*3:(k+1)*3,...] #target image x_0
            log[modality]["reconstruction"] = xrec[:, k*3:(k+1)*3, ...] #VAE reconstruction of image input without diffusion in latent space
        
            if plot_diffusion_rows: #computes steps of forward process and logs it
                # get diffusion row
                diffusion_row = list()
                z_start = z[:n_row, k*4:(k+1)*4, ...]
                for t in range(self.num_timesteps):
                    if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                        t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                        t = t.to(self.device).long()
                        noise = torch.randn_like(z_start)
                        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                        if modality != "optical_flow":
                            diffusion_row.append(self.decode_first_stage(z_noisy))
                        else:
                            diffusion_row.append(self.first_stage_model_flow.decode((1/self.scale_factor) * z_noisy)) # TODO faire proprement

                diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
                diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
                diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
                diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
                log[modality]["diffusion_row"] = diffusion_grid

            
            if sample: #sampling of the conditional diffusion model with DDIM for accelerated inference without logging intermediates
                samples_m = samples[:, k*4:(k+1)*4,...]
                if modality != "optical_flow":
                    x_samples = self.decode_first_stage(samples_m) #decodes generated samples to image space
                else:
                    x_samples = self.first_stage_model_flow.decode((1/self.scale_factor) * samples_m) #decodes generated samples to image space
                    log[modality]["correspondence_weights"] = repeat(batch["correspondence_weights"], "b h w -> b c h w", c=3)

                log[modality]["samples"] = x_samples
                if plot_denoise_rows:
                    denoise_grid = self._get_denoise_row_from_list(z_denoise_row, modality=modality) #a remplacer avec flow
                    log[modality]["denoise_row"] = denoise_grid

                if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                        self.first_stage_model, IdentityFirstStage):
                    # also display when quantizing x0 while sampling
                    # TODO
                    raise NotImplemented("Logging quantize_denoised sampling not implemented for DiffMap ")
                    # with ema_scope("Plotting Quantized Denoised"):
                    #     samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                    #                                             ddim_steps=ddim_steps,eta=ddim_eta,
                    #                                             quantize_denoised=True)
                    #     # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #     #                                      quantize_denoised=True)
                    # x_samples = self.decode_first_stage(samples.to(self.device))
                    # log["samples_x0_quantized"] = x_samples

            if unconditional_guidance_scale > 1.0: #sampling with classifier free guidance
                samples_cfg_m = samples_cfg[: , k*4:(k+1)*4, ...]
                if modality != "optical_flow":
                    x_samples_cfg = self.decode_first_stage(samples_cfg_m)
                else:
                    x_samples_cfg = self.first_stage_model_flow.decode((1/self.scale_factor) * samples_cfg_m)
                log[modality][f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

            if inpaint:
                # TODO
                raise NotImplemented("Logging inpaint sampling not implemented for DiffMap ")
                # samples_inpaint_m = samples_inpaint[: , k*4:(k+1)*4, ...]
                # x_samples = self.decode_first_stage(samples_inpaint_m.to(self.device))
                # log[modality]["samples_inpainting"] = x_samples
                # log[modality]["mask"] = mask     
                # samples_outpaint_m =  samples_outpaint[: , k*4:(k+1)*4, ...]       
                # x_samples = self.decode_first_stage(samples_outpaint_m.to(self.device))
                # log[modality]["samples_outpainting"] = x_samples

            if plot_progressive_rows:
                progressives_m = [s[:, k*4:(k+1)*4, ...] for s in progressives]
                prog_row = self._get_denoise_row_from_list(progressives_m, desc="Progressive Generation", modality=modality)
                log[modality][f"progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log[self.modalities[0]].keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {modality:{key: log[key] for key in return_keys} for modality in self.modalities}
        return log
    
    def shared_step(self, batch, **kwargs):
        z, c, flows, flows_masks, correspondence_weights  = self.get_input(batch, self.first_stage_key, return_flows_depths=True)
        loss = self(z, c, flows, flows_masks, correspondence_weights)
        return loss

    def forward(self, z, c, flows, flows_masks, correspondence_weights, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(z, c, flows, flows_masks, correspondence_weights, t, *args, **kwargs)

    def p_losses(self, z_start, cond, flows, flows_masks, correspondence_weights, t, noise=None):
        # Prepares input for U-Net diffusion
        noise = default(noise, lambda: torch.randn_like(z_start))
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)

        # Computes model output and predicts clean x_0
        model_output = self.apply_model(z_noisy, t, cond) #prediction of x_0 or noise from x_t depending on the parameterization
        if self.parameterization == "x0":
            target = z_start
            z_recon = model_output
        elif self.parameterization == "eps":
            target = noise
            z_recon = self.predict_start_from_noise(z_noisy, t=t, noise=model_output) #x_0 estimated from the noise estimated and x_t:=x
        else:
            raise NotImplementedError()
        
        # Prepares flowmap inputs
        x_recon_flowmap = self.decode_first_stage_all(z_recon, modalities=["depth_trgt", "depth_ctxt"])
        dummy_flowmap_batch, flows, depths_recon, correspondence_weights = self.get_input_flowmap(x_recon_flowmap, flows, flows_masks, correspondence_weights)

        #computes losses for every modality
        loss_dict = {}
        prefix = 'train' if self.training else 'val'        
        loss_simple, loss_gamma, loss, loss_vlb  = 0, 0, 0, 0
        logvar_t = self.logvar[t].to(self.device)
        if self.learn_logvar:
            loss_dict.update({'logvar': self.logvar.data.mean()})
        
        for k in range(len(self.modalities)):
            modality = self.modalities[k]

            if modality ==  "depth_ctxt": #flowmap loss
                loss_flowmap = self.flowmap_loss_wrapper(dummy_flowmap_batch, flows, depths_recon, correspondence_weights, self.global_step)
                loss_m = loss_flowmap
                loss_dict.update({f'{prefix}_flowmap/loss': loss_flowmap.clone().detach()})
                loss += loss_m
            elif modality == "depth_trgt":
                pass #TODO remove hack and properly handle modalities
            else: #diffusion losses
                pass
                # loss_simple_m = self.get_loss(model_output[:, k*4:(k+1)*4, ...], target[:, k*4:(k+1)*4, ...], mean=False).mean([1, 2, 3])
                # loss_simple += loss_simple_m
                # loss_dict.update({f'{prefix}_{modality}/loss_simple': loss_simple_m.clone().detach().mean()})

                # loss_gamma_m = loss_simple_m / torch.exp(logvar_t) + logvar_t
                # if self.learn_logvar:
                #     loss_dict.update({f'{prefix}_{modality}/loss_gamma': loss_gamma_m.mean()})
                # loss_gamma += loss_gamma_m

                # loss_vlb_m = self.get_loss(model_output[:, k*4:(k+1)*4, ...], target[:, k*4:(k+1)*4, ...], mean=False).mean(dim=(1, 2, 3))
                # loss_vlb_m = (self.lvlb_weights[t] * loss_vlb_m).mean()
                # loss_dict.update({f'{prefix}_{modality}/loss_vlb': loss_vlb_m})
                # loss_vlb += loss_vlb_m

                # loss_m = self.l_simple_weight * loss_gamma_m.mean() + (self.original_elbo_weight * loss_vlb_m)
                # loss_dict.update({f'{prefix}_{modality}/loss': loss_m})

                # loss += loss_m

        # loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
        # if self.learn_logvar:
        #     loss_dict.update({f'{prefix}/loss_gamma': loss_gamma.mean()})
        # loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        
        # print("GRADIENTS convin_0 mean : ", gradients_convin0.mean())
        # print("GRADIENTS convin_1 mean : ", gradients_convin1.mean())

        if prefix == "train":
            try:
                gradient_in = self.model.diffusion_model.input_blocks[0][0].weight._grad.abs().mean().clone().detach()
                loss_dict.update({f'{prefix}/l1_gradient_convin': gradient_in})
                # print("GRADIENTS convin_0.0.weight mean : ",  self.model.diffusion_model.input_blocks[0][0].weight._grad.mean())
                # print("GRADIENTS convin_0.0.bias mean : ",  self.model.diffusion_model.input_blocks[0][0].bias._grad.mean())
            except Exception:
                pass
            
            try:
                gradient_out = self.model.diffusion_model.out[2].weight._grad.abs().mean().clone().detach()
                loss_dict.update({f'{prefix}/l1_gradient_weight_convout': gradient_out})
                # print("GRADIENTS conv_out.weight mean : ",  self.model.diffusion_model.out[2].weight._grad.mean())
                # print("GRADIENTS conv_out.bias mean : ", self.model.diffusion_model.out[2].bias._grad.mean())
            except Exception:
                pass
                # print("OUTPUT DIDN'T RECEIVE GRAD")
                # print("OUTPUT WEIGHT, BIAS SIZE : ", self.model.diffusion_model.out[2].weight.size(), self.model.diffusion_model.out[2].bias.size())
                # print("OUTPUT WEIGHT, BIAS REQUIRE GRAD? : ", self.model.diffusion_model.out[2].weight.requires_grad, self.model.diffusion_model.out[2].bias.requires_grad)


        # z0 = self.split_modalities(z_recon, modalities=['depth_ctxt', 'depth_trgt'])
        # z0_depth_ctxt, z0_depth_trgt = z0['depth_ctxt'], z0['depth_trgt']
        # x0_depth_ctxt, x0_depth_trgt = x_recon_flowmap['depth_ctxt'], x_recon_flowmap['depth_trgt']
        # model_output_split = self.split_modalities(model_output, modalities=['depth_ctxt', 'depth_trgt'])
        # model_output_depth_ctxt, model_output_depth_trgt = model_output_split['depth_ctxt'], model_output_split['depth_trgt']
        
        
        # model_output.register_hook(lambda grad: print("GRAD MEAN MODEL OUTPUT :", grad.mean()))
        # z_recon.register_hook(lambda grad: print("GRAD MEAN ESTIMATED Z0 :", grad.mean()))
        # x_recon_flowmap['depth_trgt'].register_hook(lambda grad: print("GRAD MEAN X0_DEPTH_TRGT :", grad.mean()))
        # x_recon_flowmap['depth_ctxt'].register_hook(lambda grad: print("GRAD MEAN X0_DEPTH_CTXT :", grad.mean()))
        # depths_recon.register_hook(lambda grad: print("GRAD MEAN X0_DEPTHS FLOWMAP INPUT :", grad.mean()))
        # z0_depth_ctxt.register_hook(lambda grad: print("GRAD MEAN Z0 ESTIMATED DEPTH_CTXT :", grad.mean()))
        # z0_depth_trgt.register_hook(lambda grad: print("GRAD MEAN Z0 ESTIMATED DEPTH_TRGT :", grad.mean()))
        # x0_depth_ctxt.register_hook(lambda grad: print("GRAD MEAN X0 ESTIMATED DEPTH_CTXT  :", grad.mean()))
        # x0_depth_trgt.register_hook(lambda grad: print("GRAD MEAN X0 ESTIMATED DEPTH_TRGT :", grad.mean()))
        # model_output_depth_ctxt.register_hook(lambda grad: print("GRAD MEAN MODEL OUTPUT DEPTH_CTXT  :", grad.mean()))
        # model_output_depth_trgt.register_hook(lambda grad: print("GRAD MEAN MODEL OUTPUT DEPTH_TRGT :", grad.mean()))


        
        loss_dict.update({f'{prefix}/loss': loss})
    
        return loss, loss_dict
    
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.unet_trainable in ["attn", "conv_in", "all", True]:
            return super().configure_optimizers()
        elif self.unet_trainable == "conv_out":
            print("Training only unet output conv layers")
            params.extend(list(self.model.diffusion_model.out[2].parameters()))
        elif self.unet_trainable == "conv_io":
            print("Training only unet input and output conv layers")
            params.extend(list(self.model.diffusion_model.input_blocks[0][0].parameters()))
            params.extend(list(self.model.diffusion_model.out[2].parameters()))
        elif self.unet_trainable == "conv_io_attn":
            print("Training unet input, output conv and cross-attention layers")
            params.extend(list(self.model.diffusion_model.input_blocks[0][0].parameters()))
            params.extend(list(self.model.diffusion_model.out[2].parameters()))
            for n, m in self.model.named_modules():
                if isinstance(m, CrossAttention) and n.endswith('attn2'):
                    params.extend(m.parameters())
        else:
            raise ValueError(f"Unrecognised setting for unet_trainable: {self.unet_trainable}")

        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)

        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt
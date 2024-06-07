from jaxtyping import Int
from omegaconf import DictConfig

from .ddpm import *
from .diffusion_wrapper import DiffusionMapWrapper
from ldm.modules.flowmap.flow import Flows
from ldm.models.diffusion.ddim import DDIMSamplerDiffmap
from ldm.models.diffusion.types import Sample, DiffusionOutput
from ldm.misc.modalities import Modalities

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


class DDPMDiffmap(DDPM):
    """main class"""
    def __init__(
        self,
        cond_stage_config: DictConfig,
        modalities_config: ListConfig,
        num_timesteps_cond: int | None = None,
        cond_stage_key: str = "image", #key in batches (:=dicts) for the conditioning signal (eg could be images, text…)
        cond_stage_trainable: bool = False,
        concat_mode: bool = True,
        cond_stage_forward = None,
        conditioning_key = None, #defines the type of conditioning (cross attention, concatenation, hybrid)
        unet_trainable: bool = True,
        n_future: int = 1,
        flowmap_loss_config: DictConfig | None = None,
        compute_weights: bool = False,
        *args,
        **kwargs
    ) -> None:

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        assert self.num_timesteps_cond <= kwargs['timesteps']

        # For backwards compatibility with class DiffusionMapWrapper.
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        # Instantiate modalities.
        self.modalities_in = Modalities(modalities_config.modalities_in)
        self.modalities_out = Modalities(modalities_config.modalities_out)
        assert len(set([modality.channels_m for modality in self.modalities_in.modality_list])) <= 1, "Found different number of channels in input modalities."
        assert len(set([modality.channels_m for modality in self.modalities_out.modality_list])) <= 1, "Found different number of channels in output modalities."
        assert self.modalities_out.modality_list[0].channels_m == self.modalities_in.modality_list[0].channels_m, "Input and output modalities should have same individual channel counts."
        self.channels_m = self.modalities_out.modality_list[0].channels_m

        # Instantiate model.
        assert self.modalities_out.n_noisy_channels == self.modalities_in.n_noisy_channels
        kwargs['channels'] = self.modalities_in.n_noisy_channels #enforce number of noisy channels for model
        model = DiffusionMapWrapper(
            kwargs['unet_config'],
            conditioning_key, kwargs['image_size'],
            modalities_out=self.modalities_out,
            compute_weights=compute_weights,
            n_future=n_future,
        ) #U-Net
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs, model=model)

        # Instantiate conditioning encoder.
        self.cond_stage_trainable = cond_stage_trainable
        self.unet_trainable = unet_trainable
        self.cond_stage_key = cond_stage_key
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        # Load checkpoint.
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # Diffmap specific
        self.n_future = n_future
        # Running without diffusion.
        if self.channels == 0:
            assert self.parameterization == "x0", "No denoising mode only allowed with x0 parameterization mode"
            assert self.model.conditioning_key in ['concat', 'hybrid'], "No input modalities or conditioning for U-Net"

        # Instantiate flowmap loss.
        if flowmap_loss_config is not None:
            self.flowmap_loss_wrapper = self.instantiate_flowmap_loss(flowmap_loss_config)

        # Enable not training - only viz. TODO remove?
        if self.unet_trainable is False:
            self.model_ema.m_name2s_name = {}

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()  #feature extractor pour conditioneur?
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model #CLIP for conditioning

    def instantiate_flowmap_loss(self, cfg_dict) -> FlowmapLossWrapper:
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
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.unet_trainable == "attn":
            print("Training only unet attention layers")
            for n, m in self.model.named_modules():
                if isinstance(m, CrossAttention) and n.endswith('attn2'):
                    params.extend(m.parameters())
        if self.unet_trainable == "conv_in":
            print("Training only unet input conv layers")
            params = list(self.model.diffusion_model.input_blocks[0][0].parameters())
        elif self.unet_trainable is True or self.unet_trainable == "all":
            print("Training the full unet")
            params = list(self.model.parameters())
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
        elif self.unet_trainable is False:
            params = list(self.model.parameters())
            for p in params:
                p.requires_grad = False
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
    
    def _get_denoise_row_from_list(self, samples, desc='', modality=None):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(zd.to(self.device))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        if modality in ['depth_ctxt', 'depth_trgt']:
            denoise_grid = self.to_depth(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_input_modality(self, batch, k) -> Float[Tensor, "sample (frame channel) height width"]:
        x = batch[k]
        if len(x.shape) == 4:
            x = x[..., None]
        x = rearrange(x, 'b f h w c -> b (f c) h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    @torch.no_grad()
    def get_input(self, batch, k, force_c_encode=False,
                    cond_key=None, return_original_cond=False, bs=None, return_flows_depths=False):
        """
        Returns the inputs and outputs required for a diffusion sampling process and to supervise the model.
        Inputs:
            - k: string, self.first_stage_key, key for target image x_0 in the batch
        Output:
            - (default): list [x, c] with x target image x_0, c conditioning signal
            - if force_c_encode: c is the feature encoding of the conditioning signal (eg with CLIP)
            - if return_original_cond: adds xc, conditioning signal (non-encoded)
        """
        if cond_key is None: #mostly the case
                cond_key = self.cond_stage_key

        # Get input modalities.
        if len(self.modalities_in) > 0:
            x_list = list()
            for modality in self.modalities_in.modality_list:
                #encodes target image in VAE latent space
                x = self.get_input_modality(batch, modality.name) #image target clean x_0
                if bs is not None:
                    x = x[:bs]
                x = x.to(self.device)
                x_list.append(x)
            x = torch.cat(x_list, dim=1)
        else: #only conditioning as input
            b, _, h, w = self.get_input_modality(batch, cond_key).size()
            if bs is not None:
                b = min(b, bs)
            x = torch.zeros((b, 0, h, w), device=self.device)

        # Get conditioning image.
        if self.model.conditioning_key is not None:
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox', "txt"]:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else: #mostly the case, cond_key is self.cond_stage_key and different from input image
                    xc = self.get_input_modality(batch, cond_key).to(self.device) #conditioning image
            else:
                xc = x
            if bs is not None:
                xc = xc[:bs]

            # Encode conditioning image.
            if (not self.cond_stage_trainable or force_c_encode) and self.model.conditioning_key != "concat":
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
        # No conditioning.        
        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        
        #outputs
        out = [x, c] #x target image x_0, c encoding for conditioning signal
        if return_flows_depths: #TODO properly handle modalities
            bs = default(bs, x.size(0))
            bs = min(bs, x.size(0))
            correspondence_weights = batch['correspondence_weights'][:bs,:,:,:].to(self.device)
            flows = Flows(**{
                "forward": batch['optical_flow'][:bs, :, :, :, :2].to(self.device),
                "backward":  batch['optical_flow_bwd'][:bs, :, :, :, :2].to(self.device),
                "forward_mask": batch['optical_flow_mask'][:bs, :, :, :].to(self.device),
                "backward_mask": batch['optical_flow_bwd_mask'][:bs, :, :, :].to(self.device)
                }
            )
            out.extend([flows, correspondence_weights])
        if return_original_cond:
            out.append(xc)
        return out

    def get_input_flowmap(
            self,
            depths: Float[Tensor, "sample frame channels height width"],
            flows: Flows,
            correspondence_weights: Float[Tensor, "sample pair height width"]
        ) -> tuple[dict[str, Tensor], dict[str, Tensor], Float[Tensor, "sample frame height width"], Float[Tensor, "sample pair height width"]]:
        
        # Prepare depths.
        depths = self.to_depth(depths)
        # TODO - properly handle modalities
        depths = self.modalities_out.split_modalities_multiplicity(depths, modality_names=["depth_ctxt", "depth_trgt"])
        depths = torch.cat([
            depths["depth_ctxt"],
            depths["depth_trgt"]
            ],
            dim=1
        )

        # flows = {
        #     "forward": flows.forward,
        #     "backward": flows.backward,
        #     "forward_mask": flows.forward_mask,
        #     "backward_mask": flows.backward_mask,
        # }
        # flows = Flows(**flows)
    
        # if 'camera_ctxt' in batch and 'camera_trgt' in batch:
        #     intrinsics_ctxt = torch.stack([cam.K for cam in batch['camera_ctxt']], dim=0).float() #(frame 3 3)
        #     intrinsics_trgt = torch.stack([cam.K for cam in batch['camera_trgt']], dim=0).float() #(frame 3 3)
        #     intrinsics = torch.stack([intrinsics_ctxt, intrinsics_trgt], dim=1).to(self.device) #(frame pair=1 3 3)
        #     dummy_flowmap_batch['intrinsics'] = intrinsics

        # Prepare flowmap dummy batch - TODO remove hack
        B, _ , H, W, _ = flows.forward.size()
        dummy_flowmap_batch = {
            "videos": torch.zeros((B, 1 + self.n_future, 3, H, W), dtype=torch.float32, device=self.device),
            "indices": repeat(torch.arange(1 + self.n_future, device=self.device), 'f -> b f', b=B),
            "scenes": [""],
            "datasets": [""],
        }

        return dummy_flowmap_batch, flows, depths, correspondence_weights

    def training_step(self, batch, batch_idx):
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1-p, p]):
                    batch[k][i] = val

        if self.unet_trainable is not False or not torch.is_grad_enabled():
            loss, loss_dict = self.shared_step(batch)
            self.log_dict(loss_dict, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True)

            self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)

            if self.use_scheduler:
                lr = self.optimizers().param_groups[0]['lr']
                self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            return loss
        # Enable not training - viz only TODO remove?
        else:
            return None

    def shared_step(self, batch, **kwargs):
        x, c, flows, correspondence_weights  = self.get_input(batch, self.first_stage_key, return_flows_depths=True)
        loss = self(x, c, flows, correspondence_weights)
        return loss
    
    def forward(self, x, c, flows, correspondence_weights, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, flows, correspondence_weights, t, *args, **kwargs)

    def p_losses(
        self,
        x_start: Float[Tensor, "sample (frame channel) height width"],
        cond: list[Float[Tensor, "sample channel height width"]],
        flows: dict,
        correspondence_weights: Float[Tensor, "sample pair=frame-1 height width"],
        t: Int[Tensor, f"sample"] ,
        noise: Float[Tensor, "sample (frame channel) height width"]| None = None
    ) -> tuple[float, dict] | None:
        """Compute diffusion and flowmap losses.
        """
        # Enable not training - viz only TODO remove?
        if not self.unet_trainable:
            assert not any(p.requires_grad for p in self.model.parameters())

        # Prepare input for U-Net diffusion.
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Perform denoising step.
        model_output = self.apply_model(x_noisy, t, cond) # denoising
        if self.parameterization == "x0":
            denoising_target = x_start
        elif self.parameterization == "eps":
            denoising_target = noise
        else:
            raise NotImplementedError()
        
        use_flowmap_loss = any([modality.channels_m * modality.multiplicity > 0 for modality in self.modalities_out.clean_modalities if modality.modality == "depth"]) 
        use_denoising_loss = self.channels > 0
        assert use_flowmap_loss or use_denoising_loss, "No loss is being optimized"

        # Prepare diffusion loss inputs.
        if use_denoising_loss:
            denoising_output = self.modalities_out.split_modalities(model_output.diff_output, modality_names=self.modalities_out.names_denoised)
            denoising_target = self.modalities_out.split_modalities(denoising_target, modality_names=self.modalities_out.names_denoised)

            denoising_output = {k:rearrange(v, 'b (f c) h w -> (b f) c h w', c=self.channels_m) for k,v in denoising_output.items()}
            denoising_target = {k:rearrange(v, 'b (f c) h w -> (b f) c h w', c=self.channels_m) for k,v in denoising_target.items()}

        # Prepare flowmap loss inputs.
        if use_flowmap_loss:
            depths = rearrange(model_output.clean, 'b (f c) h w -> b f c h w', c=self.channels_m)
            correspondence_weights = default(model_output.weights, correspondence_weights)
            dummy_flowmap_batch, flows, depths, correspondence_weights = self.get_input_flowmap(depths, flows, correspondence_weights)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'        
        loss_simple, loss_gamma, loss, loss_vlb  = 0, 0, 0, 0
        logvar_t = self.logvar.to(self.device)
        logvar_t = repeat(logvar_t[t], 'b -> b f', f=self.n_future)
        logvar_t = rearrange(logvar_t, 'b f -> (b f)')
        if self.learn_logvar:
            loss_dict.update({'logvar': self.logvar.data.mean()})

        # Compute flowmap loss.
        if use_flowmap_loss:
            loss_flowmap = self.flowmap_loss_wrapper(dummy_flowmap_batch, flows, depths, correspondence_weights, self.global_step)
            loss_m = loss_flowmap
            loss_dict.update({f'{prefix}_flowmap/loss': loss_flowmap.clone().detach()})
            loss += loss_m

        # Compute diffusion losses for every denoised modality.
        if use_denoising_loss:
            for modality in self.modalities_out.denoised_modalities:
                name_m = modality.name

                # Compute diffusion losses for modality.
                loss_simple_m = self.get_loss(denoising_output[name_m], denoising_target[name_m], mean=False).mean([1, 2, 3])
                loss_simple += loss_simple_m
                loss_dict.update({f'{prefix}_{name_m}/loss_simple': loss_simple_m.clone().detach().mean()})

                loss_gamma_m = loss_simple_m / torch.exp(logvar_t) + logvar_t
                if self.learn_logvar:
                    loss_dict.update({f'{prefix}_{name_m}/loss_gamma': loss_gamma_m.mean()})
                loss_gamma += loss_gamma_m

                loss_vlb_m = self.get_loss(denoising_output[name_m], denoising_target[name_m], mean=False).mean(dim=(1, 2, 3))
                lvlb_weights = repeat(self.lvlb_weights[t], 'b -> b f', f=self.n_future)
                lvlb_weights = rearrange(lvlb_weights, 'b f -> (b f)')
                loss_vlb_m = (lvlb_weights * loss_vlb_m).mean()
                loss_dict.update({f'{prefix}_{name_m}/loss_vlb': loss_vlb_m})
                loss_vlb += loss_vlb_m

                loss_m = self.l_simple_weight * loss_gamma_m.mean() + (self.original_elbo_weight * loss_vlb_m)
                loss_dict.update({f'{prefix}_{name_m}/loss': loss_m})

                loss += loss_m
        
            # Log total diffusion losses.
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss_gamma.mean()})
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        # Log input / output layers gradients.
        if prefix == "train":
            try:
                gradient_in = self.model.diffusion_model.input_blocks[0][0].weight._grad.abs().mean().clone().detach()
                loss_dict.update({f'{prefix}/l1_gradient_convin': gradient_in})
                # print("GRADIENTS convin_0.0.weight mean : ",  self.model.diffusion_model.input_blocks[0][0].weight._grad.mean())
                # print("GRADIENTS convin_0.0.bias mean : ",  self.model.diffusion_model.input_blocks[0][0].bias._grad.mean())
            except Exception:
                pass
            
            try:
                gradient_out = self.model.diff_out[2].weight._grad.abs().mean().clone().detach()
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
        # Enable not training - viz only TODO remove?
        if self.unet_trainable is not False or not torch.is_grad_enabled():
            return loss, loss_dict
        else:
            return None

    def apply_model(
        self,
        x_noisy: Float[Tensor, "sample (frame channel) height width"],
        t: Int[Tensor, f"sample"] ,
        cond: list[Float[Tensor, "sample channel height width"]]
    ) -> DiffusionOutput:
        """Prepare denoising step inputs and perform denoising step.
        """
        # Prepare conditioning.
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
        # Denoising step.
        out = self.model(x_noisy, t, **cond)
        return out

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    #estimates posterior q(x_{t-1}|x_t, x_0, c) mean and variance given x_t and conditioning c at step t
    def p_mean_variance(
        self,
        samples: Sample,
        c,
        t: int,
        clip_denoised: bool,
        score_corrector=None,
        corrector_kwargs=None
    ) -> tuple[
        Float[Tensor, 'batch channel height width'],
        Float[Tensor, 'batch channel height width'],
        Float[Tensor, 'batch channel height width'],
        Sample
    ]:
        t_in = t
        x_noisy = samples.x_noisy

        # Estimate x_recon=:x_0 at step t from x_noisy:=x_t and conditioning c.
        model_output = self.apply_model(x_noisy, t_in, c) # U-Net output
        diff_output = model_output.diff_output
        if score_corrector is not None:
            assert self.parameterization == "eps"
            diff_output = score_corrector.modify_score(self, diff_output, x_noisy, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=diff_output)
        elif self.parameterization == "x0":
            x_recon = diff_output
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # Estimate mean and variance at step t from x_t and estimated x_0.
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_noisy, t=t)

        # Update x0 estimate for this step.
        samples.x_recon = x_recon

        # Compute and store depths and weights.
        if self.modalities_out.n_clean_channels > 0:
            depths = rearrange(model_output.clean, 'b (f c) h w -> b f c h w', c=self.channels_m)
            samples.depths = self.to_depth(depths)
            samples.weights = model_output.weights

        return model_mean, posterior_variance, posterior_log_variance, samples

    #takes x_t and conditioning c as input at step t, returns denoised x_{t-1}
    @torch.no_grad()
    def p_sample(self, samples: Sample, c, t, clip_denoised=False, repeat_noise=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None
                 ) -> Sample:
        x_noisy = samples.x_noisy
        b, *_, device = *x_noisy.shape, x_noisy.device
        model_mean, _, model_log_variance, samples = self.p_mean_variance(samples=samples, c=c, t=t, clip_denoised=clip_denoised,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

        noise = noise_like(x_noisy.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_noisy.shape) - 1)))

        # Denoise x_{t-1} sampled (step 4 sampling algo DDPM).
        x_denoised = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Update sample.
        samples.x_noisy = x_denoised
        samples.t = t
        return samples

    #methode qui definit la MC pour sampler, legerement diff de p_sample_loop
    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None) -> tuple[Sample, list[Float[Tensor, '...']]]:
        
        # Prepare sampling initialization.
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        
        # Initialize sampling.
        img = torch.randn(shape, device=self.device) if x_T is None else x_T
        samples = Sample(**{"x_noisy": img, "t": timesteps})

        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        #sample backward process
        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            samples = self.p_sample(samples, cond, ts,
                                            clip_denoised=self.clip_denoised, temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs) #iteration, diff than for p_sample_loop
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                samples.x_noisy = img_orig * mask + (1. - mask) * samples.x_noisy

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(samples.x_recon)
            if callback: callback(i)
            if img_callback: img_callback(samples.x_noisy, i)
        return samples, intermediates

    #samples Monte Carlo backward sampling chain
    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, mask=None,
                      x0=None, img_callback=None, start_T=None, log_every_t=None) -> Sample | tuple[Sample, list[Float[Tensor, '...']]]:

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]

        # Prepare sampling initialization.
        img = torch.randn(shape, device=device) if x_T is None else x_T
        timesteps = default(timesteps, self.num_timesteps)
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        samples = Sample(**{"x_noisy": img, "t": timesteps})
        intermediates = [img]

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        iterator = tqdm(reversed(range(0, timesteps)),desc='Sampling t', total=timesteps) if verbose\
            else reversed(range(0, timesteps))
        
        # Monte-Carlo sampling by interating on inverse timesteps.
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            # Forward process for conditioning if it is noised.
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            # Samples x_{t-1} with a diffusion step.
            samples = self.p_sample(samples, cond, ts,
                                clip_denoised=self.clip_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                samples.x_noisy = img_orig * mask + (1. - mask) * samples.x_noisy

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(samples.x_noisy)
            if callback: callback(i)
            if img_callback: img_callback(samples.x_noisy, i)

        if return_intermediates:
            return samples, intermediates
        return samples

    #methode pour sampler le DDPM, generation
    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, mask=None, x0=None, shape=None,**kwargs) -> Sample:
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, mask=mask, x0=x0) # Monte Carlo sampling for backward process

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs) -> tuple[Sample, list[Float[Tensor, '...']]]:
        # If diffusion model gets at least an input.
        if self.modalities_in.n_channels > 0:
            if ddim:
                ddim_sampler = DDIMSamplerDiffmap(self)
                shape = (self.channels, self.image_size, self.image_size)
                samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                            shape, cond, verbose=False, **kwargs)

            else:
                samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                    return_intermediates=True, **kwargs)
        # If no input for diffusion model - only conditioning.
        else:
            b, _, h, w = cond.size()
            dummy_x_noisy = torch.zeros((b, 0, h, w), device=self.device)
            intermediates = torch.zeros((b, 0, h, w), device=self.device)
            t = torch.randint(0, self.num_timesteps, (cond.shape[0],), device=self.device).long()
            model_output = self.apply_model(dummy_x_noisy, t, cond)
            depths = rearrange(model_output.clean, 'b (f c) h w -> b f c h w', c=self.channels_m)
            depths = self.to_depth(depths)
            samples = Sample(dummy_x_noisy, 0, x_recon=None, depths=depths, weights=model_output.weights)

        return samples, intermediates
    
    @torch.no_grad()
    def sample_intermediate(
        self,
        x_start: Float[Tensor, "sample noisy_channel height width"],
        cond,
        t
    ) -> Float[Tensor, "sample viz=4 frame channel height width"]:
        """Visualize intermediate diffusion step.
        """
        # Intermediate input.
        t_intermediate = torch.full((x_start.shape[0],), t, device=self.device, dtype=torch.long)
        x_T_intermediate = self.q_sample(x_start=x_start, t=t_intermediate)
        samples_intermediate = Sample(**{"x_noisy": x_T_intermediate, "t": t})        

        # Sample diffusion step.
        samples_intermediate = self.p_sample(samples_intermediate, cond, t_intermediate, clip_denoised=self.clip_denoised)

        out = dict()
        tensor_vizs = [x_start, x_T_intermediate, samples_intermediate.x_noisy, samples_intermediate.x_recon]

        # Split noisy modalities.
        for k in range(len(tensor_vizs)):
            x = rearrange(tensor_vizs[k], 'b (f c) h w -> b f c h w', c=self.channels_m)
            tensor_vizs[k] = self.modalities_out.split_modalities_multiplicity(x, modality_names=self.modalities_out.names_denoised)

        # Gather visualizations.
        for name_m in self.modalities_out.names_denoised:
            out[name_m] = torch.stack([tensor_vizs[k][name_m] for k in range(len(tensor_vizs))], dim=1)

        return out

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        if null_label is not None:
            xc = null_label
            if isinstance(xc, ListConfig):
                xc = list(xc)
            if isinstance(xc, dict) or isinstance(xc, list):
                c = self.get_learned_conditioning(xc)
            else:
                if hasattr(xc, "to"):
                    xc = xc.to(self.device)
                c = self.get_learned_conditioning(xc)
        else:
            # todo: get null label from cond_stage_model
            raise NotImplementedError()
        c = repeat(c, '1 ... -> b ...', b=batch_size).to(self.device)
        return c

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    def to_depth(self, x: Float[Tensor, "sample frame 3 height width"]) -> Float[Tensor, "sample frame height width"]:
        depths = x.mean(2)
        return (depths / 1000).exp() + 0.01
    
    @torch.no_grad()
    def log_images(
        self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., inpaint=True,
        plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True,
        unconditional_guidance_scale=1., unconditional_guidance_label=None, use_ema_scope=True, **kwargs
    ) -> dict:
        """Sample diffusion model and create visualizations for logging.
        """
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None and self.num_timesteps > 1

        # Prepare inputs
        log = dict()
        x, c, xc = self.get_input(
            batch,
            self.first_stage_key, force_c_encode=True,return_original_cond=True,
            bs=N
        )
        
        x_log = rearrange(x, 'b (f c) h w -> b f c h w', c=self.channels_m)
        xc_log = rearrange(xc, 'b (f c) h w -> b f c h w', c=self.channels_m)
        x_split = self.modalities_in.split_modalities_multiplicity(x_log)

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        
        # Log conditioning images.
        if self.model.conditioning_key is not None:
            log['conditioning'] = dict()
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"][self.model.conditioning_key] = xc
            elif self.cond_stage_key in ["caption", "txt"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch[self.cond_stage_key], size=x.shape[2]//25)
                log["conditioning"][self.model.conditioning_key] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"], size=x.shape[2]//25)
                log["conditioning"][self.model.conditioning_key] = xc
            elif isimage(xc):
                log["conditioning"][self.model.conditioning_key] = xc_log
            if ismap(xc):
                log["conditioning"][self.model.conditioning_key] = self.to_rgb(xc)

        # Log input modalities.
        for name_m in self.modalities_in.names:
            log[name_m] = dict()
            log[name_m]["inputs"] = x_split[name_m]

            if plot_diffusion_rows: #computes steps of forward process and logs it
                assert False, "not implemented with n_future frames"
                # get diffusion row
                diffusion_row = list()
                x_start = x[name_m][:n_row]
                for t in range(self.num_timesteps):
                    if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                        t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                        t = t.to(self.device).long()
                        noise = torch.randn_like(x_start)
                        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                        diffusion_row.append(x_noisy)

                diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
                diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
                diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
                diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
                log[name_m]["diffusion_row"] = diffusion_grid

        # Sample diffusion model.
        if sample:
            # get denoise row
            with ema_scope("Sampling"):
                samples, x_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)
            # Format samples.
            weights = default(samples.weights, batch["correspondence_weights"])
            samples_diffusion = rearrange(samples.x_noisy, 'b (f c) h w -> b f c h w', c=self.channels_m)
            samples_diffusion = self.modalities_out.split_modalities_multiplicity(samples_diffusion, modality_names=self.modalities_out.names_denoised)
            # TODO - properly handle modalities
            samples_depths = self.modalities_out.split_modalities_multiplicity(samples.depths, modality_names=["depth_ctxt", "depth_trgt"])
            samples = dict(samples_diffusion, **samples_depths)

            # Visualize intermediate diffusion step.
            if self.modalities_out.n_noisy_channels > 0:
                t_intermediate = self.num_timesteps // 2
                samples_intermediate = self.sample_intermediate(x_start=x, cond=c, t=t_intermediate)

        # Sampling with classifier-free guidance.
        if unconditional_guidance_scale > 1.0 and self.model.conditioning_key not in ["concat", "hybrid"]:
            uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # uc = torch.zeros_like(c)
            with ema_scope("Sampling with classifier-free guidance"):
                samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta,
                                                 unconditional_guidance_scale=unconditional_guidance_scale,
                                                 unconditional_conditioning=uc,
                                )
                samples_cfg = rearrange(samples_cfg.x_noisy, 'b (f c) h w -> b f c h w', c=self.channels_m)
                samples_cfg = self.modalities_out.split_modalities_multiplicity(samples_cfg, modality_names=self.modalities_out.names_denoised)

        if inpaint:
            # make a simple center square
            b, h, w = x.shape[0], x.shape[2], x.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):

                samples_inpaint, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=x[:N], mask=mask)
                samples_inpaint = samples_inpaint.x_noisy
                
            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples_outpaint, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=x[:N], mask=mask)
                samples_outpaint = samples_outpaint.x_noisy
        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                _, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
                
                for k in range(len(progressives)):
                    s = rearrange(s, 'b (f c) h w -> b f c h w', c=self.channels_m)
                    progressives[k] = self.modalities_out.split_modalities_multiplicity(s, modality_names=self.modalities_out.names_denoised)

        # Log samples.
        for name_m in self.modalities_out.names:
            if name_m not in log.keys():
                log[name_m] = dict()

            if sample:
                x_samples = samples[name_m]
                
                # Add correspondence weights and GT depths - TODO do properly by handling modalities.
                if name_m in ['depth_trgt', 'depth_ctxt']:
                    depths = self.get_input_modality(batch, name_m)[:N]
                    depths = rearrange(depths, 'b (f c) h w -> b f c h w', c=self.channels_m)
                    log[name_m]["inputs"] = depths
                    if name_m == 'depth_trgt':
                        log[name_m]["correspondence_weights"] = weights
                
                # Log samples.
                log[name_m]["samples"] = x_samples

                # Add intermediates diffusion logs.
                if self.channels > 0:
                    if name_m in samples_intermediate.keys():
                        log[name_m][f"intermediates_{t_intermediate}"] = samples_intermediate[name_m]
                        
                if plot_denoise_rows and self.channels > 0:
                    denoise_grid = self._get_denoise_row_from_list(x_denoise_row, modality=name_m) #to replace with flow
                    log[name_m]["denoise_row"] = denoise_grid

            # Log classifier-free guidance samples.
            if unconditional_guidance_scale > 1.0 and self.model.conditioning_key not in ["concat", "hybrid"]:
                x_samples_cfg = samples_cfg[name_m]
                log[name_m][f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

            if inpaint:
                # TODO
                raise NotImplemented("Logging inpaint sampling not implemented for DiffMap ")
                # samples_inpaint_m = samples_inpaint[: , k*4:(k+1)*4, ...]
                # x_samples = self.decode_first_stage(samples_inpaint_m.to(self.device))
                # log[name_m]["samples_inpainting"] = x_samples
                # log[name_m]["mask"] = mask     
                # samples_outpaint_m =  samples_outpaint[: , k*4:(k+1)*4, ...]       
                # x_samples = self.decode_first_stage(samples_outpaint_m.to(self.device))
                # log[name_m]["samples_outpainting"] = x_samples

            if plot_progressive_rows:
                progressives_m = [s[name_m] for s in progressives]
                prog_row = self._get_denoise_row_from_list(progressives_m, desc="Progressive Generation", modality=name_m)
                log[name_m][f"progressive_row"] = prog_row

        return log
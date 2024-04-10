from .ddpm import *

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


from torchvision.utils import save_image

class DDPMDiffmap(DDPM):
    """main class"""
    def __init__(self,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image", #key in batches (:=dicts) of the conditioning signal (eg could be images, text…)
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None, #defines the type of conditioning ie cross attention, concatenation, hybrid
                 unet_trainable=True,
                 modalities=['nfp'],
                 flowmap_loss_config=None,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.unet_trainable = unet_trainable
        self.cond_stage_key = cond_stage_key
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # Diffmap specific
        self.modalities = modalities
        assert self.channels % len(self.modalities) == 0, "Number of channels should be a multiple of number of modalities"
        self.channels_m = self.channels // len(self.modalities) #number of individual channels per modality

        if flowmap_loss_config is not None:
            self.flowmap_loss_wrapper = self.instantiate_flowmap_loss(flowmap_loss_config)

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

    def instantiate_flowmap_loss(self, cfg_dict):
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
    
    def split_modalities(
            self,
            x: Float[Tensor, "_ 3*C _ _"],
            C: int = None,
            modalities: list[str] | None = None
        ) -> dict[str, Float[Tensor, "_ C _ _"]]:
        # Splits input tensor along every modality of chunk size C
        C = default(C, self.channels_m)
        modalities = default(modalities, self.modalities)
        split_all = dict(zip(self.modalities, torch.split(x, C, dim=1)))
        out = {m: split_all[m] for m in modalities}
        return out

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(zd.to(self.device))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
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

    @torch.no_grad()
    def get_input(self, batch, k, force_c_encode=False,
                    cond_key=None, return_original_cond=False, bs=None, return_flows_depths=False):
        '''
        Returns the inputs and outputs required for a diffusion sampling process and to supervise the model.
        Inputs:
            - k: string, self.first_stage_key, key for target image x_0 in the batch
        Output:
            - (default): list [x, c] with x target image x_0, c conditioning signal
            - if force_c_encode: c is the feature encoding of the conditioning signal (eg with CLIP)
            - if return_original_cond: adds xc, conditioning signal (non-encoded)
        '''
        x_list = list()
        for modality in self.modalities:
            #encodes target image in VAE latent space
            x = super().get_input(batch, modality) #image target clean x_0
            if bs is not None:
                x = x[:bs]
            x = x.to(self.device)
            x_list.append(x)
        
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
                    xc = super().get_input(batch, cond_key).to(self.device) #conditioning image
            else:
                xc = x

            #encodes conditioning image with feature encoder (eg CLIP)
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

        else: #no conditioning in this case
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        
        #outputs
        out = [x, c] #x target image x_0, c encoding for conditioning signal
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
            x_recon_flowmap: dict[str, Float[Tensor, "batch channels height width"]],
            flows: dict[str, Float[Tensor, "batch channels height width"]],
            flows_masks: dict[str, Float[Tensor, "batch height width"]],
            correspondence_weights: Float[Tensor, "batch height width"]
        ) -> tuple[dict[str, Tensor], dict[str, Tensor], Float[Tensor, "batch frame height width"]]:
        # Prepare depth, should be (batch frame height width).
        correspondence_weights = correspondence_weights[:, None, :, :]
        depths_recon = torch.stack([
            x_recon_flowmap["depth_ctxt"].mean(1),
            x_recon_flowmap["depth_trgt"].mean(1)
            ],
            dim=1
        )
        # Normalize the depth with min - max.
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

    def shared_step(self, batch, **kwargs):
        x, c, flows, flows_masks, correspondence_weights  = self.get_input(batch, self.first_stage_key, return_flows_depths=True)
        loss = self(x, c, flows, flows_masks, correspondence_weights)
        return loss
    
    def forward(self, x, c, flows, flows_masks, correspondence_weights, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, flows, flows_masks, correspondence_weights, t, *args, **kwargs)

    def p_losses(self, x_start, cond, flows, flows_masks, correspondence_weights, t, noise=None):
        # Prepares input for U-Net diffusion
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Computes model output and predicts clean x_0
        model_output = self.apply_model(x_noisy, t, cond) #prediction of x_0 or noise from x_t depending on the parameterization
        if self.parameterization == "x0":
            target = x_start
            x_recon = model_output
        elif self.parameterization == "eps":
            target = noise
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=model_output) #x_0 estimated from the noise estimated and x_t:=x
        else:
            raise NotImplementedError()
        
        # Prepares flowmap inputs
        x_recon_flowmap = self.split_modalities(x_recon, modalities=["depth_trgt", "depth_ctxt"])
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
                # loss_simple_m = self.get_loss(model_output[:, k*3:(k+1)*3, ...], target[:, k*3:(k+1)*3, ...], mean=False).mean([1, 2, 3])
                # loss_simple += loss_simple_m
                # loss_dict.update({f'{prefix}_{modality}/loss_simple': loss_simple_m.clone().detach().mean()})

                # loss_gamma_m = loss_simple_m / torch.exp(logvar_t) + logvar_t
                # if self.learn_logvar:
                #     loss_dict.update({f'{prefix}_{modality}/loss_gamma': loss_gamma_m.mean()})
                # loss_gamma += loss_gamma_m

                # loss_vlb_m = self.get_loss(model_output[:, k*3:(k+1)*3, ...], target[:, k*3:(k+1)*3, ...], mean=False).mean(dim=(1, 2, 3))
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

    # Prepare U-Net inputs and call U-Net.
    def apply_model(self, x_noisy, t, cond):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)  #applies the U-Net, DiffusionWrapper

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

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
    def p_mean_variance(self, x, c, t, clip_denoised: bool,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t

        #estimates x_recon=:x_0 at step t from x_t and conditioning c
        model_out = self.apply_model(x, t_in, c) # U-Net output
        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)


        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        #estimates mean and variance at step t from x_t and estimated x_0
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    #takes x_t and conditioning c as input at step t, returns denoised x_{t-1}
    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        #denoised x_{t-1} (step 4 sampling alg DDPM)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    #methode qui definit la MC pour sampler, legerement diff de p_sample_loop
    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
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

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs) #iteration, diff than for p_sample_loop
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    #samples Monte Carlo backward sampling chain
    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, mask=None,
                      x0=None, img_callback=None, start_T=None, log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        #itere sur les timesteps en temps inverse
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond)) #~processus forward, noise conditionnement? pourquoi?

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised) #methode qui definit une iteration du sampling du DDPM
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    #methode pour sampler le DDPM, generation
    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, mask=None, x0=None, shape=None,**kwargs):
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
                                  verbose=verbose, timesteps=timesteps, mask=mask, x0=x0) #methode qui definit la MC pour sampler

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                         shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True, **kwargs)

        return samples, intermediates

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
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True, plot_diffusion_rows=True,
                   unconditional_guidance_scale=1., unconditional_guidance_label=None, use_ema_scope=True, **kwargs):
        '''
        Performs diffusion forward and backward process on a given batch and returns logs for tensorboard
        '''
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None and self.num_timesteps > 1

        log = dict()
        x, c, xc = self.get_input(batch, self.first_stage_key,
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
                    samples, x_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                            ddim_steps=ddim_steps,eta=ddim_eta) #samples generative process in latent space
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)

        #sampling with classifier free guidance
        if unconditional_guidance_scale > 1.0 and self.model.conditioning_key not in ["concat", "hybrid"]:
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
            b, h, w = x.shape[0], x.shape[2], x.shape[3]
            mask = torch.ones(N, h, w).to(self.device)
            # zeros will be filled in
            mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
            mask = mask[:, None, ...]
            with ema_scope("Plotting Inpaint"):

                samples_inpaint, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=x[:N], mask=mask)
                
            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples_outpaint, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=x[:N], mask=mask)
        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)


        for k in range(len(self.modalities)):
            modality = self.modalities[k]
            log[modality] = dict()
            log[modality]["inputs"] = x[:, k*3:(k+1)*3,...] #target image x_0
        
            if plot_diffusion_rows: #computes steps of forward process and logs it
                # get diffusion row
                diffusion_row = list()
                x_start = x[:n_row, k*3:(k+1)*3, ...]
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
                log[modality]["diffusion_row"] = diffusion_grid

            
            if sample: #sampling of the conditional diffusion model with DDIM for accelerated inference without logging intermediates
                x_samples = samples[:, k*3:(k+1)*3,...]
                if modality != "optical_flow":
                    log[modality]["correspondence_weights"] = repeat(batch["correspondence_weights"], "b h w -> b c h w", c=3)

                log[modality]["samples"] = x_samples
                if plot_denoise_rows:
                    denoise_grid = self._get_denoise_row_from_list(x_denoise_row, modality=modality) #a remplacer avec flow
                    log[modality]["denoise_row"] = denoise_grid

            if unconditional_guidance_scale > 1.0 and self.model.conditioning_key not in ["concat", "hybrid"]: #sampling with classifier free guidance
                x_samples_cfg = samples_cfg[: , k*3:(k+1)*3, ...]
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
                progressives_m = [s[:, k*3:(k+1)*3, ...] for s in progressives]
                prog_row = self._get_denoise_row_from_list(progressives_m, desc="Progressive Generation", modality=modality)
                log[modality][f"progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log[self.modalities[0]].keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {modality:{key: log[key] for key in return_keys} for modality in self.modalities}
        return log


    
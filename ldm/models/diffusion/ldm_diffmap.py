from .ddpm import *
from .unet_wrapper import DiffusionMapWrapper

from ldm.thirdp.flowmap.flowmap.flow import Flows
from ldm.models.diffusion.ddim import DDIMSamplerDiffmap
from ldm.models.diffusion.types import Sample, DiffusionOutput


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}



class FlowmapLDM(LatentDiffusion): #derived from LatentInpaintDiffusion
    """
    can either run as pure inpainting model (only concat mode) or with mixed conditionings,
    e.g. mask as concat and text via cross-attn.
    To disable finetuning mode, set finetune_keys to None

    Simultaneous diffusion of flow, depths and next frame
    """
    def __init__(self,
                *args,
                first_stage_flow_config=None,
                modalities_in=['nfp'],
                modalities_out=['nfp'],
                flowmap_loss_config=None,
                compute_weights=False,
                **kwargs):
        model = DiffusionMapWrapper(kwargs['unet_config'], kwargs['conditioning_key'], kwargs['image_size'], compute_weights=compute_weights, latent=True) #U-Net
        super().__init__(model=model, *args, **kwargs)

        # Diffmap specifics.
        if first_stage_flow_config is not None:
            model = instantiate_from_config(first_stage_flow_config)
            self.first_stage_model_flow = model.eval()  #encodeur-decoder VAE
            self.first_stage_model_flow.train = disabled_train
            for param in self.first_stage_model_flow.parameters():
                param.requires_grad = False

        # Diffmap specific
        self.modalities_in = modalities_in
        self.modalities_out = modalities_out
        assert self.model.diffusion_model.out_channels % len(self.modalities_out) == 0, "Number of output channels should be a multiple of number of output modalities"
        self.channels_m = self.model.diffusion_model.out_channels // len(self.modalities_out)

        if len(self.modalities_in) > 0:
            assert self.channels % len(self.modalities_in) == 0,  "Number of input channels should be a multiple of number of input modalities"
            assert self.channels_m == self.model.diffusion_model.out_channels // len(self.modalities_out), "Number of channels for every modality should match be equal for input and output"
        else: #case where only conditioning as input
            assert self.parameterization == "x0", "No-denoising mode only allowed with x0 parameterization mode"
            assert self.model.conditioning_key in ['concat', 'hybrid'], "No input modalities or conditioning for U-Net"

        if flowmap_loss_config is not None:
            self.flowmap_loss_wrapper = self.instantiate_flowmap_loss(flowmap_loss_config)
            
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
        
    def split_modalities(
            self,
            z: Float[Tensor, "_ M*C _ _"],
            modalities: list[str],
            C: int = None,
        ) -> dict[str, Float[Tensor, "_ C _ _"]]:
        # Splits input tensor along every modality of chunk size C
        C = default(C, self.channels_m)
        assert C*len(modalities) == z.size(1), f"Tried splitting a tensor along dim 1 of size {z.size(1)} in {len(modalities)} chunks of size {C}"
        
        return dict(zip(modalities, torch.split(z, C, dim=1)))
    
    def decode_first_stage_modality(
            self,
            z_m: Float[Tensor, "_ C _ _"],
            modality: str,
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
            modalities: list[str]
        ) -> dict[str, Float[Tensor, "_ 4 _ _"]]:
        # Decodes input modalities
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

        if cond_key is None: #mostly the case
                cond_key = self.cond_stage_key

        # Get input modalities.
        if len(self.modalities_in) > 0:
            x_list, z_list = list(), list()
            for modality in self.modalities_in:
                #encodes target image in VAE latent space
                x = DDPM.get_input(self, batch, modality) #image target clean x_0
                if bs is not None:
                    x = x[:bs]
                x = x.to(self.device)
                if modality != "optical_flow":
                    encoder_posterior = self.encode_first_stage(x)  #encode image target clean, latent clean encodé E(x_0)
                    z = self.get_first_stage_encoding(encoder_posterior).detach() #sample espace latent VAE (dans ce cas, juste encoder_posterior scalé)
                else:
                    # TODO do it properly
                    encoder_posterior = self.first_stage_model_flow.encode(x)
                    z = self.scale_factor * encoder_posterior.sample()

                x_list.append(x)
                z_list.append(z)
            
            z = torch.cat(z_list, dim=1)
            x = torch.cat(x_list, dim=1)
        else: #only conditioning as input
            b, _, h, w = DDPM.get_input(self, batch, cond_key).size()
            x = torch.zeros((b, 0, h, w), device=self.device)
            # TODO do it properly
            x_dummy = torch.zeros((b, 3, h, w), device=self.device)
            encoder_posterior_dummy = self.encode_first_stage(x_dummy)
            _, _, h_z, w_z  = encoder_posterior_dummy.sample().size()
            z = torch.zeros((b, 0, h_z, w_z), device=self.device)
        

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
                else: #mostly the case, cond_key is self.cond_stage_key and different from input image
                    xc = DDPM.get_input(self, batch, cond_key).to(self.device) #conditioning image
            else:
                xc = x

            #encodes conditioning image with feature encoder (eg CLIP)
            if self.model.conditioning_key == "concat":
                encoder_posterior_c = self.encode_first_stage(xc)
                c = self.get_first_stage_encoding(encoder_posterior_c).detach()
                if bs is not None:
                    c = c[:bs]
            else:
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
            if len(self.modalities_in) > 0:
                x_rec_split = self.decode_first_stage_all(z, modalities=self.modalities_in)
                xrec = torch.cat([x_rec_split[modality] for modality in self.modalities_in], dim=1)
            else:
                b, _, h, w = DDPM.get_input(self, batch, cond_key).size()
                xrec = torch.zeros((b, 0, h, w), device=self.device)

            out.extend([x, xrec])
        if return_x:
            out.append(x)
        if return_flows_depths: #TODO properly handle modalities
            bs = default(bs, x.size(0))
            bs = min(bs, x.size(0))
            correspondence_weights = batch['correspondence_weights'][:bs,None,:,:].to(self.device)
            flows = Flows(**{
                "forward": batch['optical_flow'][:bs, None, :, :, :2].to(self.device),
                "backward":  batch['optical_flow_bwd'][:bs, None, :, :, :2].to(self.device),
                "forward_mask": batch['optical_flow_mask'][:bs, None, :, :].to(self.device),
                "backward_mask": batch['optical_flow_bwd_mask'][:bs, None, :, :].to(self.device)
                }
            )
            out.extend([flows, correspondence_weights])
        if return_original_cond:
            out.append(xc)
        return out

    def get_input_flowmap(
            self,
            depths: dict[str, Float[Tensor, "batch channels height width"]],
            flows: Flows,
            correspondence_weights: Float[Tensor, "batch pair height width"]
        ) -> tuple[dict[str, Tensor], dict[str, Tensor], Float[Tensor, "batch frame height width"], Float[Tensor, "batch pair height width"]]:
        # Prepare depth, should be (batch frame height width).
        # correspondence_weights = correspondence_weights[:, None, :, :]
        # Compute depth with exponential mapping.
        depths = torch.cat([
            self.to_depth(depths["depth_ctxt"]),
            self.to_depth(depths["depth_trgt"])
            ],
            dim=1
        )

        # # Prepare flow
        # # flows_recon = rearrange(x_recon_flowmap["optical_flow"][:, None, :2, :, :], 'b f xy w h -> b f w h xy') #estimated clean forward flow, TODO, should be (batch pair height width 2)
        # flows_fwd = flows["forward"][:, None, :, :, :2]  #gt clean forward flows, TODO, should be (batch pair height width 2)
        # flows_bwd = flows["backward"][:, None, :, :, :2]  #gt clean backward flows, TODO, should be (batch pair height width 2)
        # flows_mask_fwd = flows_masks["forward"][:, None, :, :] #gt clean forward flows consistency masks, TODO, should be (batch pair height width)
        # flows_mask_bwd = flows_masks["backward"][:, None, :, :] #gt clean backward flows consistency masks, TODO, should be (batch pair height width)
        
        # flows = {
        #     "forward": flows_fwd,
        #     "backward": flows_bwd,
        #     "forward_mask": flows_mask_fwd,
        #     "backward_mask": flows_mask_bwd,
        # }

        # Prepare flowmap dummy batch TODO remove hack
        N, _ ,_, H, W = flows.forward.size()
        dummy_flowmap_batch = {
            "videos": torch.zeros((N, 2, 3, H, W), dtype=torch.float32, device=self.device),
            "indices": torch.tensor([0,1], device=self.device).repeat(N, 2),
            "scenes": [""],
            "datasets": [""],
        }

        return dummy_flowmap_batch, flows, depths, correspondence_weights
    
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
        if modality in ['depth_ctxt', 'depth_trgt']:
            denoise_grid = self.to_depth(denoise_grid)
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid
                
    def shared_step(self, batch, **kwargs):
        z, c, flows, correspondence_weights  = self.get_input(batch, self.first_stage_key, return_flows_depths=True)
        loss = self(z, c, flows, correspondence_weights)
        return loss

    def forward(self, z, c, flows, correspondence_weights, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(z, c, flows, correspondence_weights, t, *args, **kwargs)

    def p_losses(self, z_start, cond, flows, correspondence_weights, t, noise=None):
        # Prepare input for U-Net diffusion.
        noise = default(noise, lambda: torch.randn_like(z_start))
        z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)

        # Compute model output and predicts clean x_0.
        model_output = self.apply_model(z_noisy, t, cond) #prediction of x_0 or noise from x_t depending on the parameterization
        if self.parameterization == "x0":
            denoising_target = z_start
        elif self.parameterization == "eps":
            denoising_target = noise
        else:
            raise NotImplementedError()
        
        # Prepare loss inputs.
        use_flowmap_loss = model_output.clean.size(1) > 0
        use_denoising_loss = model_output.diff_output.size(1) > 0
        assert use_flowmap_loss or use_denoising_loss, "No loss is optimized"
        if use_denoising_loss:
            denoising_output = self.split_modalities(model_output.diff_output, self.modalities_in)
            denoising_target = self.split_modalities(denoising_target, self.modalities_in)

        # Compute losses for every modality.
        loss_dict = {}
        prefix = 'train' if self.training else 'val'        
        loss_simple, loss_gamma, loss, loss_vlb  = 0, 0, 0, 0
        logvar_t = self.logvar.to(self.device)
        logvar_t = logvar_t[t]
        if self.learn_logvar:
            loss_dict.update({'logvar': self.logvar.data.mean()})
        
        for modality in self.modalities_out:
            
            # Flowmap loss.
            if modality ==  "depth_ctxt" and use_flowmap_loss:
                # Prepare flowmap inputs.
                depths = self.decode_first_stage_all(model_output.clean, modalities=["depth_ctxt", "depth_trgt"])
                correspondence_weights = default(model_output.weights, correspondence_weights)
                # correspondence_weights = torch.cat([correspondence_weights]*4, dim=1)
                # correspondence_weights = 1. / self.scale_factor * correspondence_weights
                # correspondence_weights = self.first_stage_model.decode(correspondence_weights).mean(dim=1, keepdims=True)
                dummy_flowmap_batch, flows, depths, correspondence_weights = self.get_input_flowmap(depths, flows, correspondence_weights)

                # Compute flowmap loss.
                loss_flowmap = self.flowmap_loss_wrapper(dummy_flowmap_batch, flows, depths, correspondence_weights, self.global_step)
                loss_m = loss_flowmap
                loss_dict.update({f'{prefix}_flowmap/loss': loss_flowmap.clone().detach()})
                loss += loss_m
 
            elif modality == "depth_trgt": #TODO remove hack and properly handle modalities
                pass
            elif use_denoising_loss: #diffusion losses
                # Compute diffusion loss.
                loss_simple_m = self.get_loss(denoising_output[modality], denoising_target[modality], mean=False).mean([1, 2, 3])
                loss_simple += loss_simple_m
                loss_dict.update({f'{prefix}_{modality}/loss_simple': loss_simple_m.clone().detach().mean()})

                loss_gamma_m = loss_simple_m / torch.exp(logvar_t) + logvar_t
                if self.learn_logvar:
                    loss_dict.update({f'{prefix}_{modality}/loss_gamma': loss_gamma_m.mean()})
                loss_gamma += loss_gamma_m

                loss_vlb_m = self.get_loss(denoising_output[modality], denoising_target[modality], mean=False).mean(dim=(1, 2, 3))
                loss_vlb_m = (self.lvlb_weights[t] * loss_vlb_m).mean()
                loss_dict.update({f'{prefix}_{modality}/loss_vlb': loss_vlb_m})
                loss_vlb += loss_vlb_m

                loss_m = self.l_simple_weight * loss_gamma_m.mean() + (self.original_elbo_weight * loss_vlb_m)
                loss_dict.update({f'{prefix}_{modality}/loss': loss_m})

                loss += loss_m
        if use_denoising_loss:
            loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})
            if self.learn_logvar:
                loss_dict.update({f'{prefix}/loss_gamma': loss_gamma.mean()})
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        if prefix == "train":
            try:
                gradient_in = self.model.diffusion_model.input_blocks[0][0].weight._grad.abs().mean().clone().detach()
                loss_dict.update({f'{prefix}/l1_gradient_convin': gradient_in})
            except Exception:
                pass
            
            try:
                gradient_out = self.model.diff_out[2].weight._grad.abs().mean().clone().detach()
                loss_dict.update({f'{prefix}/l1_gradient_weight_convout': gradient_out})
            except Exception:
                pass
        
        loss_dict.update({f'{prefix}/loss': loss})
    
        return loss, loss_dict
    
    #fonction dans laquelle est formellement appele le U-Net
    def apply_model(self, z_noisy, t, cond) -> DiffusionOutput:

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]            
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(z_noisy, t, **cond)  #applies the U-Net, DiffusionWrapper
        return x_recon
    
    #estimates posterior q(x_{t-1}|x_t, x_0, c) mean and variance given x_t and conditioning c at step t
    def p_mean_variance(self, samples: Sample,
                        c,
                        t: int,
                        clip_denoised: bool,
                        quantize_denoised=False,
                        score_corrector=None,
                        corrector_kwargs=None
                        ) -> tuple[Float[Tensor, 'batch channel height width'],
                                   Float[Tensor, 'batch channel height width'],
                                   Float[Tensor, 'batch channel height width'],
                                   Sample
                                   ]:
        t_in = t
        z_noisy = samples.x_noisy

        # Estimate x_recon=:x_0 at step t from z_noisy:=z_t and conditioning c.
        model_output = self.apply_model(z_noisy, t_in, c) # U-Net output
        diff_output = model_output.diff_output
        if score_corrector is not None:
            assert self.parameterization == "eps"
            diff_output = score_corrector.modify_score(self, diff_output, z_noisy, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(z_noisy, t=t, noise=diff_output)
        elif self.parameterization == "x0":
            x_recon = diff_output
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)

        # Estimate mean and variance at step t from x_t and estimated x_0.
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=z_noisy, t=t)

        # Update x0 estimate for this step.
        samples.x_recon = x_recon

        # Compute and store depths and weights.
        if model_output.clean.size(1) > 0:
            # TODO do it cleanly by handling modalities in a secure way
            depths = self.decode_first_stage_all(model_output.clean, ["depth_ctxt", "depth_trgt"])
            weights = model_output.weights
            # weights = torch.cat([weights]*4, dim=1)
            # weights = 1. / self.scale_factor * weights
            # weights = self.first_stage_model.decode(weights).mean(dim=1, keepdims=True)

            depths = torch.cat([
                self.to_depth(depths["depth_ctxt"]),
                self.to_depth(depths["depth_trgt"])
                ],
                dim=1
            )
            samples.depths = depths
            samples.weights = weights

        return model_mean, posterior_variance, posterior_log_variance, samples
    
    #takes x_t and conditioning c as input at step t, returns denoised x_{t-1}
    @torch.no_grad()
    def p_sample(self, samples: Sample, c, t, clip_denoised=False, repeat_noise=False,
                 quantize_denoised=False, temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None) -> Sample:
        z_noisy = samples.x_noisy
        b, *_, device = *z_noisy.shape, z_noisy.device
        model_mean, _, model_log_variance, samples = self.p_mean_variance(samples=samples, c=c, t=t, clip_denoised=clip_denoised,
                                       quantize_denoised=quantize_denoised,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)

        noise = noise_like(z_noisy.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(z_noisy.shape) - 1)))

        # Denoise x_{t-1} sampled (step 4 sampling algo DDPM).
        z_denoised = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Update sample.
        samples.x_noisy = z_denoised
        samples.t = t
        return samples
    
    #methode qui definit la MC pour sampler, legerement diff de p_sample_loop
    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
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
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
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
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None) -> Sample | tuple[Sample, list[Float[Tensor, '...']]]:

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

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        # Monte-Carlo sampling by interating on inverse timesteps.
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            # Forward process for conditioning if it is noised.
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond)) #~processus forward, noise conditionnement? pourquoi?

            # Samples x_{t-1} with a diffusion step.
            samples = self.p_sample(samples, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised) #methode qui definit une iteration du sampling du DDPM
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                samples.x_noisy = img_orig * mask + (1. - mask) * samples.x

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(samples.x_noisy)
            if callback: callback(i)
            if img_callback: img_callback(samples.x_noisy, i)

        if return_intermediates:
            return samples, intermediates
        return samples
    
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs) -> tuple[Sample, list[Float[Tensor, '...']]]:
        if len(self.modalities_in) > 0:
            if ddim:
                ddim_sampler = DDIMSamplerDiffmap(self)
                shape = (self.channels, self.image_size, self.image_size)
                samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size,
                                                            shape, cond, verbose=False, **kwargs)

            else:
                samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                    return_intermediates=True, **kwargs)
                
        else:
            b, _, h_z, w_z = cond.size()
            dummy_z_noisy = torch.zeros((b, 0, h_z, w_z), device=self.device)
            intermediates = torch.zeros((b, 0, h_z, w_z), device=self.device)
            t = torch.randint(0, self.num_timesteps, (cond.shape[0],), device=self.device).long()
            model_output = self.apply_model(dummy_z_noisy, t, cond)
            depths = self.decode_first_stage_all(model_output.clean, ["depth_ctxt", "depth_trgt"])
            depths = torch.cat([
                self.to_depth(depths["depth_ctxt"]),
                self.to_depth(depths["depth_trgt"])
                ],
                dim=1
            )
            samples = Sample(dummy_z_noisy, 0, x_recon=None, depths=depths, weights=model_output.weights)


        return samples, intermediates
    
    def to_depth(self, x: Float[Tensor, "batch 3 height width"]) -> Float[Tensor, "batch 1 height width"]:
        depths = x.mean(1, keepdims=True)
        return (depths / 1000).exp() + 0.01
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        if self.unet_trainable in ["attn", "conv_in", "all", True]:
            return super().configure_optimizers()
        elif self.unet_trainable == "conv_out":
            print("Training only unet output conv layers")
            params.extend(list(self.model.diff_out[2].parameters()))
        elif self.unet_trainable == "conv_io":
            print("Training only unet input and output conv layers")
            params.extend(list(self.model.diffusion_model.input_blocks[0][0].parameters()))
            params.extend(list(self.model.diff_out[2].parameters()))
        elif self.unet_trainable == "conv_io_attn":
            print("Training unet input, output conv and cross-attention layers")
            params.extend(list(self.model.diffusion_model.input_blocks[0][0].parameters()))
            params.extend(list(self.model.diff_out[2].parameters()))
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
    
    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1.,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, unconditional_guidance_scale=1., unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        '''
        Performs diffusion forward and backward process on a given batch and returns logs for tensorboard
        '''
        ema_scope = self.ema_scope if use_ema_scope else nullcontext
        use_ddim = ddim_steps is not None and self.num_timesteps > 1

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        x_split = self.split_modalities(x, modalities=self.modalities_in, C=3)
        x_rec_split = self.split_modalities(xrec, modalities=self.modalities_in, C=3)

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
                log["conditioning"][self.model.conditioning_key] = xc
            if ismap(xc):
                log["conditioning"][self.model.conditioning_key] = self.to_rgb(xc)

        # Log input modalities.
        for modality in self.modalities_in:
            log[modality] = dict()
            log[modality]["inputs"] = x_split[modality]
            log[modality]["reconstruction"] = x_rec_split[modality] #VAE reconstruction of image input without diffusion in latent space

            if plot_diffusion_rows: #computes steps of forward process and logs it
                # get diffusion row
                diffusion_row = list()
                x_start = x[modality][:n_row]
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
                # get denoise row
                with ema_scope("Sampling"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                            ddim_steps=ddim_steps,eta=ddim_eta) #samples generative process in latent space
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
                    weights = batch["correspondence_weights"].squeeze(1) if samples.weights is None else samples.weights.squeeze(1)
                    samples_diffusion = self.decode_first_stage_all(samples.x_noisy, self.modalities_in)
                    samples_depths = self.split_modalities(samples.depths, ['depth_ctxt', 'depth_trgt'], C=1) if samples.depths is not None else {}
                    samples = dict(samples_diffusion, **samples_depths)

        #sampling with classifier free guidance
        if unconditional_guidance_scale > 1.0:
            raise NotImplemented("Logging conditional scale guidance not yet implemented for DiffMap ")
            # uc = self.get_unconditional_conditioning(N, unconditional_guidance_label)
            # # uc = torch.zeros_like(c)
            # with ema_scope("Sampling with classifier-free guidance"):
            #     samples_cfg, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
            #                                      ddim_steps=ddim_steps, eta=ddim_eta,
            #                                      unconditional_guidance_scale=unconditional_guidance_scale,
            #                                      unconditional_conditioning=uc,
            #                     )
            #     weights_cfg = batch["correspondence_weights"].squeeze(1) if samples_cfg.weights is None else samples_cfg.weights.squeeze(1)
            #     samples_cfg = self.decode_first_stage_all(samples_cfg.x_noisy, self.modalities_in)
            #     samples_diffusion_cfg = self.decode_first_stage_all(samples_cfg.x_noisy, self.modalities_in)
            #     samples_depth_cfg = self.split_modalities(samples_cfg.depths, ['depth_ctxt', 'depth_trgt'], C=1) if samples_cfg.depths is not None else {}
            #     samples_cfg = dict(samples_diffusion_cfg, **samples_depth_cfg)

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
                samples_inpaint = self.decode_first_stage(samples_inpaint.x_noisy)
                
            # outpaint
            mask = 1. - mask
            with ema_scope("Plotting Outpaint"):
                samples_outpaint, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                samples_outpaint = self.decode_first_stage(samples_outpaint.x_noisy)
        if plot_progressive_rows:
            with ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
                progressives = [self.decode_first_stage_all(s, self.modalities_in) for s in progressives]

        # Log outputs.
        for modality in self.modalities_out:
            if modality not in log.keys():
                log[modality] = dict()

            if sample:
                x_samples = samples[modality]
                if modality == 'depth_trgt':
                    log[modality]["correspondence_weights"] = weights

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
                # samples_cfg_m = samples_cfg[modality]
                # if modality != "optical_flow":
                #     x_samples_cfg = self.decode_first_stage(samples_cfg_m)
                # else:
                #     x_samples_cfg = self.first_stage_model_flow.decode((1/self.scale_factor) * samples_cfg_m)
                # log[modality][f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg
                raise NotImplemented("Logging conditional scale guidance not yet implemented for DiffMap ")
                log[modality][f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = samples_cfg[modality]
                if modality == 'depth_trgt':
                    log[modality][f"correspondence_weights_cfg_scale_{unconditional_guidance_scale:.2f}"] = weights_cfg

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
                progressives_m = [s[modality] for s in progressives]
                prog_row = self._get_denoise_row_from_list(progressives_m, desc="Progressive Generation", modality=modality)
                log[modality][f"progressive_row"] = prog_row

        return log
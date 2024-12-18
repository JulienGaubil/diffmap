from jaxtyping import Int
from omegaconf import DictConfig
from typing import Dict

from .ddpm import *
from ldm.thirdp.flowmap.flowmap.flow import Flows
from ldm.models.diffusion.ddim import DDIMSamplerDiffmap
from ldm.models.diffusion.types import Sample
from ldm.misc.modalities import Modalities, GeometryModalities

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
        cond_stage_trainable: bool = False,
        cond_stage_forward = None,
        unet_trainable: bool = True,
        n_future: int = 1,
        n_ctxt: int = 1,
        wrapper_cfg: DictConfig | None = None,
        losses_config: DictConfig | ListConfig | list | None = None,
        *args,
        **kwargs
    ) -> None:

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        assert self.num_timesteps_cond <= kwargs['timesteps']

        # Instantiate modalities.
        self.modalities_in = Modalities(modalities_config.modalities_in) if modalities_config.modalities_in is not None else Modalities([])
        self.modalities_out = Modalities(modalities_config.modalities_out)

        assert len(set([modality.channels_m for modality in self.modalities_in.modality_list])) <= 1, "Found different number of channels in input modalities, should be all equal."
        assert len(set([modality.channels_m for modality in self.modalities_out.modality_list])) <= 1, "Found different number of channels in output modalities."
        assert self.modalities_out.modality_list[0].channels_m == self.modalities_in.modality_list[0].channels_m, "Input and output modalities should have same individual channel counts."
        self.channels_m = self.modalities_out.modality_list[0].channels_m
        self.output_geometry = any(isinstance(subset, GeometryModalities) for subset in self.modalities_out.subsets)

        # For backwards compatibility with class DDPM.
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])

        # Instantiate diffusion model.
        if wrapper_cfg is not None:
            model = instantiate_from_config(
                wrapper_cfg,
                modalities_in=self.modalities_in,
                modalities_out=self.modalities_out,
                diffusion_module=self
            )
            unet_config = None
        else:
            model = None
            assert kwargs.get('unet_config', None) is not None
            unet_config = kwargs.pop('unet_config')

        super().__init__(unet_config=unet_config, *args, **kwargs, model=model)

        # Instantiate conditioning encoder.
        self.cond_stage_trainable = cond_stage_trainable
        self.unet_trainable = unet_trainable
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
        self.n_ctxt = n_ctxt
        # Running without diffusion.
        if self.modalities_in.n_noisy_channels == 0:
            assert self.parameterization == "x0", "No denoising mode only allowed with x0 parameterization mode"
            assert self.model.conditioning_key in ['concat', 'hybrid'], "No input modalities or conditioning for U-Net"

        # Instantiate losses.
        if isinstance(losses_config, DictConfig):
            self.losses = nn.ModuleList([instantiate_from_config(losses_config)])
        elif isinstance(losses_config, (ListConfig, list)):
            self.losses = nn.ModuleList([instantiate_from_config(loss_config) for loss_config in losses_config])
        else:
            self.losses = nn.ModuleList()
        assert len(self.losses) > 0, "No loss is being optimized"
        for loss in self.losses:
            loss.to(self.device)

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
        raise NotImplementedError()
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
    def get_input(
        self,
        batch: dict,
        input_ids: list[str] | str | None = None,
        force_c_encode: bool = False,
        return_original_cond: bool = False,
        bs: int = None, #batch size
        return_flows: bool = False
    ) -> tuple[Float[Tensor, "batch channel height width"] | Flows]:
        """Returns the inputs and outputs required for a diffusion sampling process and to supervise the model.
        Output:
            - (default): list [x, c] with x target image x_0, c conditioning signal
            - if force_c_encode: c is the feature encoding of the conditioning signal (eg with CLIP)
            - if return_original_cond: adds xc, conditioning signal (non-encoded)
        """

        # Prepare ids.
        if isinstance(input_ids, str):
            input_ids = [input_ids]
        if input_ids is not None:
            if isinstance(input_ids, str):
                input_ids = [input_ids]
            noisy_modality_ids = sorted(list(set([id for id in input_ids if id in self.modalities_in.ids_denoised])))
            noisy_modality_ids = sorted(list(set([id for id in input_ids if id in self.modalities_in.ids_clean])))
        else:
            noisy_modality_ids = self.modalities_in.ids_denoised
            clean_modality_ids = self.modalities_in.ids_clean
        assert len(noisy_modality_ids) + len(clean_modality_ids) > 0, 'No input and conditioning for model.'

        # Get noisy input modalities.
        x = list()
        for modality_id in noisy_modality_ids:
            x_m = self.get_input_modality(batch, modality_id) #image target clean x_0
            bs = min(default(bs, x_m.size(0)), x_m.size(0))
            x_m = x_m[:bs]
            x.append(x_m)
        if len(x) == 0: #only conditioning as input
            for modality_id in clean_modality_ids:
                b, _, h, w = self.get_input_modality(batch, modality_id).size()
                bs = min(default(bs, b), b)
                x.append(torch.zeros((bs, 0, h, w), device=self.device))
        x = torch.cat(x, dim=1).to(self.device)

        # Get clean conditioning modalities.
        c, xc = list(), list()
        if len(clean_modality_ids) > 0:
            for id in clean_modality_ids:
                x_m = self.get_input_modality(batch, id)
                x_m = x_m[:bs].to(self.device)
                
                # Encode conditioning image.
                if (not self.cond_stage_trainable or force_c_encode) and self.model.conditioning_key != "concat":
                    if isinstance(x_m, dict) or isinstance(x_m, list):
                        c_m = self.get_learned_conditioning(x_m)
                    else:
                        c_m = self.get_learned_conditioning(x_m.to(self.device)) #encoded conditioning
                else: # conditioning image not encoded
                    c_m = x_m
                xc.append(x_m)
                c.append(c_m)

            xc = torch.cat(xc, dim=1)
            c = torch.cat(c, dim=1)
        else: # unconditional model
            c = None
            xc = None
        
        # Return model inputs.
        out = [x, c] # clean image x_0, and conditioning signal

        if return_flows: # add flows without scaling
            assert all([k in batch.keys() for k in ['fwd_flow', 'bwd_flow', 'mask_fwd_flow', 'mask_bwd_flow']])
            flow_normalization = batch.get('flow_normalization', torch.ones(bs))
            flow_normalization = flow_normalization[:bs,None,None,None,None].float().to(self.device)
            flows = Flows(**{
                "forward": batch['fwd_flow'][:bs, :, :, :, :2].to(self.device) * flow_normalization,
                "backward":  batch['bwd_flow'][:bs, :, :, :, :2].to(self.device) * flow_normalization,
                "forward_mask": batch['mask_fwd_flow'][:bs, :, :, :].to(self.device),
                "backward_mask": batch['mask_bwd_flow'][:bs, :, :, :].to(self.device)
                }
            )
            out.append(flows)
        
        if return_original_cond: # add non-encoded conditioning signal
            out.append(xc)
        return out

    def get_input_losses(
        self,
        x_start: Float[Tensor, "sample (frame channel) height width"],
        cond: Float[Tensor, "sample channel height width"],
        flows: Flows,
        **kwargs
    ) -> Dict:
        """Prepare loss input batch.
        """
        noisy_input = rearrange(x_start, 'b (f c) h w -> b f c h w', c=self.channels_m)  
        clean_input = rearrange(cond, 'b (f c) h w -> b f c h w', c=self.channels_m)
        noisy_input_dict = self.modalities_in.split_modalities_multiplicity(noisy_input, dim=1, modality_ids=self.modalities_in.ids_denoised)
        clean_input_dict = self.modalities_in.split_modalities_multiplicity(clean_input, dim=1, modality_ids=self.modalities_in.ids_clean)
        input_batch = dict(
            noisy_input_dict,
            flows=flows,
            **clean_input_dict
        )

        prefix = "train" if self.training else "val"

        losses_kwargs = dict(
            batch=input_batch,
            global_step=self.global_step,
            x_start=x_start,
            prefix=prefix,
            diffusion_module=self
        )
        losses_kwargs.update(kwargs)

        return losses_kwargs

    def training_step(self, batch: Dict, batch_idx: int) -> float | None:
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

    def shared_step(self, batch, **kwargs) -> tuple[float, Dict] | None:
        if self.output_geometry:
            x, c, flows = self.get_input(batch, return_flows=True)
        else:
            x, c = self.get_input(batch)
            flows = None

        losses_kwargs = self.get_input_losses(x, c, flows)
        
        loss = self(x, c, losses_kwargs)
        return loss
    
    def forward(
        self,
        x_start: Float[Tensor, "sample (frame channel) height width"],
        cond: Float[Tensor, "sample channel height width"],
        losses_kwargs: Dict,
        *args, **kwargs
    ) -> tuple[float, dict] | None:
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=self.device).long()
        losses_kwargs['t'] = t # TODO - do it properly

        if self.model.conditioning_key is not None:
            assert cond is not None
            if self.cond_stage_trainable:
                cond = self.get_learned_conditioning(cond)
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond.float()))
        return self.p_losses(x_start, cond, t, losses_kwargs, *args, **kwargs)

    def p_losses(
        self,
        x_start: Float[Tensor, "sample (frame channel) height width"],
        cond: Float[Tensor, "sample channel height width"],
        t: Int[Tensor, f"sample"],
        losses_kwargs: Dict,
        noise: Float[Tensor, "sample (frame channel) height width"] | None = None
    ) -> tuple[float, Dict] | None:
        """Compute diffusion and flowmap losses.
        """
        loss_dict = {}
        total_loss = 0 
        prefix = "train" if self.training else "val"

        # Perform denoising step.
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        denoised, clean, weights = self.apply_model(x_noisy, t, cond) # denoising step

        losses_kwargs['noise'] = noise # TODO - remove hack

        # Split all output modalities.
        denoised = rearrange(denoised, 'b (f c) h w -> b f c h w', c=self.channels_m)
        clean = rearrange(clean, 'b (f c) h w -> b f c h w', c=self.channels_m)
        noisy_output_dict = self.modalities_out.split_modalities_multiplicity(denoised, dim=1, modality_ids=self.modalities_out.ids_denoised)
        clean_output_dict = self.modalities_out.split_modalities_multiplicity(clean, dim=1, modality_ids=self.modalities_out.ids_clean)
        output_dict = dict(noisy_output_dict, conf=weights, **clean_output_dict)

        for loss in self.losses:
            loss_value, loss_metrics_dict = loss(
                output_dict,
                self.modalities_in,
                self.modalities_out,
                **losses_kwargs
            )
            total_loss += loss_value
            loss_dict.update(loss_metrics_dict)
        loss_dict.update({f'{prefix}/loss': total_loss})

        # Log input / output layers gradients.
        if self.training:
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

            try:
                gradient_weights = self.model.corr_weighter_perpoint[4].weight._grad.abs().mean().clone().detach()
                loss_dict.update({f'{prefix}/l1_gradient_weight_correspondence': gradient_weights})
            except Exception:
                pass
        
        # Enable not training - viz only TODO remove?
        if self.unet_trainable is not False or not torch.is_grad_enabled():
            return total_loss, loss_dict
        else:
            return None

    def apply_model(
        self,
        x_noisy: Float[Tensor, "sample (frame channel) height width"],
        t: Int[Tensor, f"sample"] ,
        cond: list[Float[Tensor, "sample channel height width"]]
    ) ->  tuple[Float[Tensor, "batch channel height width"] | None]:
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
        denoised, clean, weights = self.model(x_noisy, t, **cond)
        return denoised, clean, weights

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

    def predict_start_from_denoised(
        self,
        t: int,
        x_noisy,
        x_denoised: Float[Tensor, "sample noisy_channel height width"],
        clip_denoised: bool = False
    ) -> Float[Tensor, "sample noisy_channel height width"]:
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x_noisy, t=t, noise=x_denoised)
        elif self.parameterization == "x0":
            x_recon = x_denoised
        else:
            raise NotImplementedError()
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        return x_recon

    def p_mean_variance(
        self,
        samples: Sample,
        c: Float[Tensor, "sample cond_channel height width"],
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
        """Denoising step t to estimate mean and variance of posterior distribution q(x_{t-1}|x_t, x_0, c)
        """
        t_in = t
        x_noisy = samples.x_denoised # xt

        # Denoising step and clean sample estimation.
        denoised, predicted, weights = self.apply_model(x_noisy, t_in, c) # denoising step
        if score_corrector is not None:
            assert self.parameterization == "eps"
            diff_output = score_corrector.modify_score(self, denoised, x_noisy, t, c, **corrector_kwargs)
        x_recon = self.predict_start_from_denoised(t_in, x_noisy, denoised, clip_denoised) # x0
        
        # Estimate mean and variance at step t from x_t and estimated x_0.
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x_noisy, t=t)

        # Update clean samples.
        samples.x_recon = x_recon
        samples.x_predicted = predicted
        samples.weights = weights

        return model_mean, posterior_variance, posterior_log_variance, samples

    #takes x_t and conditioning c as input at step t, returns denoised x_{t-1}
    @torch.no_grad()
    def p_sample(
        self,
        samples: Sample,
        c: dict[Float[Tensor, "sample cond_channel height width"]],
        t: int,
        clip_denoised: bool = False,
        repeat_noise: bool = False,
        temperature: float = 1.,
        noise_dropout: float = 0.,
        score_corrector=None,
        corrector_kwargs=None
    ) -> Sample:
        """Denoising step to estimate denoised sample x_{t-1} from noisy sample x_t.
        """
        # Predicts posterior mean and variance with denoising step.
        model_mean, _, model_log_variance, samples = self.p_mean_variance(
            samples=samples, c=c, t=t, clip_denoised=clip_denoised,
            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs
        )

        # Add gaussian noise for t > 0.
        x_noisy = samples.x_denoised # xt
        b, *_, device = *x_noisy.shape, x_noisy.device
        noise = noise_like(x_noisy.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_noisy.shape) - 1)))

        # Denoise x_{t-1} sampled (step 4 sampling algo DDPM).
        x_denoised = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Update denoised samples.
        samples.x_denoised = x_denoised
        samples.t = t

        return samples

    #methode qui definit la MC pour sampler, legerement diff de p_sample_loop
    @torch.no_grad()
    def progressive_denoising(
        self, cond, shape, verbose=True, callback=None, img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
        score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
        log_every_t=None
    ) -> tuple[Sample, list[Float[Tensor, '...']]]:
        
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
        samples = Sample(t=timesteps, x_denoised=img)

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

        # Sample backward process.
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
                samples.x_denoised = img_orig * mask + (1. - mask) * samples.x_denoised

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(samples.x_recon)
            if callback: callback(i)
            if img_callback: img_callback(samples.x_denoised, i)
        return samples, intermediates

    @torch.no_grad()
    def p_sample_loop(
        self, cond, shape, return_intermediates=False,
        x_T=None, verbose=True, callback=None, timesteps=None, mask=None,
        x0=None, img_callback=None, start_T=None, log_every_t=None
    ) -> Sample | tuple[Sample, list[Float[Tensor, '...']]]:
        """Monte Carlo sampling for backward process.
        """

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]

        # Prepare sampling initialization.
        img = torch.randn(shape, device=device) if x_T is None else x_T
        timesteps = default(timesteps, self.num_timesteps)
        if start_T is not None:
            timesteps = min(timesteps, start_T)
        samples = Sample(t=timesteps, x_denoised=img)
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
                samples.x_denoised = img_orig * mask + (1. - mask) * samples.x_denoised

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(samples.x_denoised)
            if callback: callback(i)
            if img_callback: img_callback(samples.x_denoised, i)

        if return_intermediates:
            return samples, intermediates
        return samples

    @torch.no_grad()
    def sample(
        self, cond, batch_size=16, return_intermediates=False, x_T=None,
        verbose=True, timesteps=None, mask=None, x0=None, shape=None,**kwargs
    ) -> Sample:
        """DDPM conditional generation.
        """
        # Prepare conditioning and noise.
        if shape is None:
            shape = (batch_size, self.modalities_in.n_noisy_channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        
        # Monte Carlo sampling for backward process.
        return self.p_sample_loop(
            cond,
            shape,
            return_intermediates=return_intermediates, x_T=x_T,
            verbose=verbose, timesteps=timesteps, mask=mask, x0=x0
        )

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs) -> tuple[Sample, list[Float[Tensor, '...']]]:
        # If diffusion model gets at least an input.
        if self.modalities_in.n_noisy_channels > 0:
            if ddim:
                ddim_sampler = DDIMSamplerDiffmap(self)
                shape = (self.modalities_in.n_noisy_channels, self.image_size, self.image_size)
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
            denoised, clean, weights = self.apply_model(dummy_x_noisy, t, cond)
            samples = Sample(t=0, x_denoised=denoised, x_predicted=clean, weights=weights)

        return samples, intermediates
    
    @torch.no_grad()
    def sample_intermediate(
        self,
        x_start: Float[Tensor, "sample noisy_channel height width"],
        cond,
        t
    ) -> tuple[Float[Tensor, "sample noisy_channel channel height width"], Sample]:
        """Visualize intermediate diffusion step.
        """
        # Intermediate input.
        t_intermediate = torch.full((x_start.shape[0],), t, device=self.device, dtype=torch.long)
        x_intermediate = self.q_sample(x_start=x_start, t=t_intermediate)
        samples_intermediate = Sample(t=t, x_denoised=x_intermediate)        

        # Sample diffusion step.
        samples_intermediate = self.p_sample(samples_intermediate, cond, t_intermediate, clip_denoised=self.clip_denoised)

        return x_intermediate, samples_intermediate

    @torch.no_grad()
    def get_unconditional_conditioning(self, batch_size, null_label=None):
        raise NotImplementedError()
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
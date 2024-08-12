import sys
sys.path.append('ldm/thirdp/dust3r')

import os
import copy
import math
import torch
import pytorch_lightning as pl
import torch.nn as nn

from typing import Dict
from jaxtyping import Int, Float
from torch import Tensor
from einops import repeat, rearrange
from omegaconf import OmegaConf, DictConfig
from functools import partial

from ldm.misc.modalities import RGBModalities
from ldm.thirdp.dust3r.dust3r.heads import head_factory
from ldm.thirdp.dust3r.dust3r.model import AsymmetricCroCo3DStereo, load_model
from ldm.misc.modalities import Modalities, RGBModalities, GeometryModalities
from ldm.thirdp.dust3r.dust3r.utils.misc import transpose_to_landscape
from ldm.thirdp.dust3r.croco.models.blocks import Block, DecoderBlock, Attention, CrossAttention, Mlp, DropPath
from ldm.thirdp.dust3r.dust3r.utils.misc import is_symmetrized, interleave, transpose_to_landscape

inf = float('inf')

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Copied from https://github.com/facebookresearch/DiT/blob/main/models.py
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class BlockTime(Block):
    """Dust3r encoding block conditioned with adaptive layer norm zero (adaLN-Zero) for noise level conditioning.
    """
    def __init__(self, dim, num_heads, croco_bloc: Block | None = None, **block_args):
        super().__init__(dim, num_heads, **block_args)

        # Load checkpoint from Croco bloc.
        if croco_bloc is not None:
            self.load_state_dict(croco_bloc.state_dict(), strict=False)

        # Adaptive Layer Norm mixing parameters regression.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )

    def forward(self, x, xpos, t):
        # Predict noise level conditioning mixing parameters.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=1)

        # Encoder self-attention.
        x = x + gate_msa.unsqueeze(1) * self.drop_path(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), xpos))
        x = x + gate_mlp.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class DecoderBlockTime(DecoderBlock):
    """Dust3r decoding block conditioned with adaptive layer norm zero (adaLN-Zero) for noise level conditioning.
    """
    def __init__(self, dim, num_heads, croco_bloc: DecoderBlock | None = None, **block_args):
        super().__init__(dim, num_heads, **block_args)

        # Load checkpoint from Croco bloc.
        if croco_bloc is not None:
            self.load_state_dict(croco_bloc.state_dict(), strict=False)

        # Adaptive Layer Norm mixing parameters regression.
        self.adaLN_modulation_x = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 8 * dim, bias=True)
        )
        self.adaLN_modulation_y = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 2 * dim, bias=True)
        )
        self.adaLN_modulation_xy = nn.Sequential(
            nn.SiLU(),
            nn.Linear(2 * dim, 1 * dim, bias=True)
        )

    def forward(self, x, y, xpos, ypos, t1, t2):
        # Predict noise level conditioning mixing parameters.
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_mca_x, scale_mca_x = self.adaLN_modulation_x(t1).chunk(8, dim=1)
        shift_mca_y, scale_mca_y = self.adaLN_modulation_y(t2).chunk(2, dim=1)
        gate_mca = self.adaLN_modulation_xy(torch.cat([t1,t2], dim=1))

        # Self and cross attention.
        x = x + gate_msa.unsqueeze(1) * self.drop_path(self.attn(modulate(self.norm1(x), shift_msa, scale_msa), xpos))
        y_ = modulate(self.norm_y(y), shift_mca_y, scale_mca_y)
        x = x + gate_mca.unsqueeze(1) * self.drop_path(self.cross_attn(modulate(self.norm2(x), shift_mca_x, scale_mca_x), y_, y_, xpos, ypos))
        x = x + gate_mlp.unsqueeze(1) * self.drop_path(self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp)))
        return x, y


class Dust3rWrapper(
    AsymmetricCroCo3DStereo,
    pl.LightningModule
):
    def __init__(
        self,
        noisy_modalities_in: Modalities,
        clean_modalities_in: Modalities,
        modalities_out: Modalities,
        dust3r_cfg: DictConfig,
        conditioning_key: str = 'crossattn',
        **kwargs
    ) -> None:
        pl.LightningModule.__init__(self)
        self.conditioning_key = conditioning_key
        self.noisy_modalities_in = noisy_modalities_in 
        self.clean_modalities_in = clean_modalities_in
        self.modalities_out = modalities_out

        # TODO - remove hack 
        assert any(
            isinstance(subset, RGBModalities) and subset._past_modality is not None and subset._past_modality.multiplicity == 1
            for subset in self.clean_modalities_in.subsets
        )
        assert any(
            isinstance(subset, RGBModalities) and subset._future_modality is not None and subset._future_modality.multiplicity == 1
            for subset in self.noisy_modalities_in.subsets
        )
        assert any(
            isinstance(subset, GeometryModalities) and subset._future_modality is not None and subset._future_modality.multiplicity == 1 \
            and subset._past_modality is not None and subset._past_modality.multiplicity == 1
            for subset in self.modalities_out.subsets
        )
        
        # If predicting geometry, instantiate with base Dust3r architecture.
        geometric_modalities = [modalities_group for modalities_group in modalities_out.subsets if isinstance(modalities_group, GeometryModalities)]
        
        # TODO - remove hacks
        assert len(geometric_modalities) <= 1, "Only one geometric modality group is supported."
        assert len(geometric_modalities) == 1
        self.geometric_modalities = geometric_modalities[0]

        if len(geometric_modalities) == 1:
            # Instantiate base Dust3r.
            dust3r_cfg = OmegaConf.to_container(dust3r_cfg.params, resolve=True)
            ckpt_path = dust3r_cfg.pop('ckpt_path')
            super().__init__(**dust3r_cfg)

            # Load from pretrained.
            if ckpt_path is not None:
                if os.path.isfile(ckpt_path):
                    self.load_checkpoint(ckpt_path, device='cpu')

            # Load noise level conditioned encoder blocs.
            for bloc in self.enc_blocks:
                bloc = BlockTime(
                    dust3r_cfg['enc_embed_dim'],
                    dust3r_cfg['enc_num_heads'],
                    bloc,
                    mlp_ratio=dust3r_cfg.get('mlp_ratio', 4.),
                    qkv_bias=True,
                    norm_layer=dust3r_cfg.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                    rope=self.rope
                )
            self.enc_blocks = nn.ModuleList(
                [
                    BlockTime(
                        dust3r_cfg['enc_embed_dim'],
                        dust3r_cfg['enc_num_heads'],
                        bloc,
                        mlp_ratio=dust3r_cfg.get('mlp_ratio', 4.),
                        qkv_bias=True,
                        norm_layer=dust3r_cfg.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                        rope=self.rope
                    )
                    for bloc in self.enc_blocks
                ]
            )
            
            # Load noise level conditioned decoder blocs.
            self.dec_blocks = nn.ModuleList(
                [
                    DecoderBlockTime(
                        dust3r_cfg['dec_embed_dim'],
                        dust3r_cfg['dec_num_heads'],
                        bloc,
                        mlp_ratio=dust3r_cfg.get('mlp_ratio', 4.),
                        qkv_bias=True,
                        norm_layer=dust3r_cfg.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                        norm_mem=dust3r_cfg.get('norm_im2_in_dec', True),
                        rope=self.rope
                    )
                    for bloc in self.dec_blocks
                ]
            )
            self.dec_blocks2 = nn.ModuleList(
                [
                    DecoderBlockTime(
                        dust3r_cfg['dec_embed_dim'],
                        dust3r_cfg['dec_num_heads'],
                        bloc,
                        mlp_ratio=dust3r_cfg.get('mlp_ratio', 4.),
                        qkv_bias=True,
                        norm_layer=dust3r_cfg.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                        norm_mem=dust3r_cfg.get('norm_im2_in_dec', True),
                        rope=self.rope
                    )
                    for bloc in self.dec_blocks2
                ]
            )

            past_geometric_head_id = '_'.join(['head', self.geometric_modalities._past_modality._id])
            future_geometric_head_id = '_'.join(['head', self.geometric_modalities._future_modality._id])
            
            setattr(self, 'downstream_' + past_geometric_head_id, copy.deepcopy(self.downstream_head1))
            setattr(self,'downstream_' + future_geometric_head_id, copy.deepcopy(self.downstream_head2))
            setattr(
                self,
                past_geometric_head_id,
                transpose_to_landscape(getattr(self, 'downstream_' + past_geometric_head_id), activate=True)
            )
            setattr(
                self,
                future_geometric_head_id,
                transpose_to_landscape(getattr(self, 'downstream_' + future_geometric_head_id), activate=True)
            )
            
            del self.downstream_head1
            del self.downstream_head2

        # Instantiate timestep embedder.
        self.t_embedder_enc = TimestepEmbedder(dust3r_cfg['enc_embed_dim'])
        self.t_embedder_dec = TimestepEmbedder(dust3r_cfg['dec_embed_dim'])
        nn.init.normal_(self.t_embedder_enc.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_enc.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder_dec.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_dec.mlp[2].weight, std=0.02)
        
        # Instantiate other downstream heads.
        for subset in modalities_out.subsets:
            if isinstance(subset, GeometryModalities):
                pass
            elif isinstance(subset, RGBModalities):
                for modality in [subset._past_modality, subset._future_modality]:
                    if modality is not None:
                        head_id = '_'.join(['head', modality._id])
                        head = head_factory('dpt', 'pts3d', self, has_conf=False)
                        head.conf_mode = None
                        head.depth_mode = ('linear', -float('inf'), float('inf'))
                        setattr(
                            self,
                            'downstream_' + head_id,
                            head
                        )
                        setattr(
                            self,
                            head_id,
                            transpose_to_landscape(getattr(self, 'downstream_' + head_id), activate=True)
                        )
            else:
                raise ValueError(f'Only RGBModalities and GeometryModalities supported, got {type(subset)}.')

    def load_checkpoint(self, ckpt_path: str, device: str = 'cpu', verbose: bool = True) -> None:
        if verbose:
            print('... loading model from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        if 'landscape_only' not in args:
            args = args[:-1] + ', landscape_only=False)'
        else:
            args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
        assert "landscape_only=False" in args
        if verbose:
            print(f"instantiating : {args}")
        s = self.load_state_dict(ckpt['model'], strict=False)
        if verbose:
            print(s)
        self.to(torch.device(device))

    def prepare_input(
        self,
        x_noisy: Float[Tensor, "sample noisy_channel height width"],
        x_clean: list[Float[Tensor, "sample clean_channel height width"]],
    ) -> tuple[Dict,Dict]:
        """Dust3r Input formatting.
        """
        x_clean = torch.cat(x_clean, 1)
        B, _, H, W = x_clean.shape

        # TODO - remove hack
        assert x_noisy.size(1) == 3 and x_clean.size(1) == 3

        true_shape1 = torch.tensor([H, W], dtype=torch.int32)
        true_shape2 = torch.tensor([H, W], dtype=torch.int32)
        true_shape1 = repeat(true_shape1, 'hw -> b hw', b=x_clean.size(0))
        true_shape2 = repeat(true_shape2, 'hw -> b hw', b=x_clean.size(0))

        view1 = dict(
            img=x_clean,
            true_shape=true_shape1,
            idx=[k for k in range(B)],
            instance=[str(k) for k in range(B)]
        )
        view2 = dict(
            img=x_noisy,
            true_shape=true_shape2,
            idx=[k for k in range(B)],
            instance=[str(k) for k in range(B)]
        )

        return view1, view2

    def prepare_output(
        self,
        dust3r_outputs: Dict[str, Float[Tensor, "sample height width channel"]]
    ) -> tuple[
        Float[Tensor, "sample noisy_channel height width"],
        Float[Tensor, "sample clean_channel height width"],
        Float[Tensor, "sample frame=2 height width"] | None,
    ]:
        """Diffmap output formatting.
        """
        # TODO - remove hack
        assert all(modality.multiplicity == 1 for modality in self.geometric_modalities)

        ccat_dict = dict()
        weights = list()
        for subset in self.modalities_out.subsets:
            for modality in [subset._past_modality, subset._future_modality]:
                if modality is not None:
                    ccat_dict[modality._id] = dust3r_outputs[modality._id]['pts3d']
                    if isinstance(subset, GeometryModalities):
                        weights.append(dust3r_outputs[modality._id]['conf'])

        # Separate noisy and clean modalities.
        diff_out = self.modalities_out.cat_modalities(ccat_dict, dim=-1)
        diff_out = rearrange(diff_out, 'b h w c -> b c h w')
        denoised, clean = self.modalities_out.split_noisy_clean(diff_out)
        weights = torch.stack(weights, dim=1) if len(weights) > 0 else None

        return denoised, clean, weights

    def _encode_image(self, image, true_shape, t):
        """Dust3r image encoding with noise level conditioning.
        """
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos, t)

        x = self.enc_norm(x)
        return x, pos, None

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, t1,t2):
        """Dust3r pair image encoding with noise level conditioning.
        """
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos, _ = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0), torch.cat([t1,t2]))
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos, _ = self._encode_image(img1, true_shape1, t1)
            out2, pos2, _ = self._encode_image(img2, true_shape2, t2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2, t1, t2):
        """Dust3r symmetric image encoding with noise level conditioning.
        """
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            raise NotImplementedError
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2], t[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2, t1, t2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)

    def _decoder(self, f1, pos1, f2, pos2, t1, t2):
        """Dust3r symmetric decoding with noise level conditioning.
        """
        final_output = [(f1, f2)]  # before projection

        # project to decoder dim
        f1 = self.decoder_embed(f1)
        f2 = self.decoder_embed(f2)

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2, t1, t2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1, t2, t1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)

    def forward(
        self,
        x: Float[Tensor, "sample (frame noisy_channel) height width"],
        t: Int[Tensor, "sample"],
        c_crossattn: list[Float[Tensor, "sample (frame clean_channel) height width"]],
        **kwargs
    ) -> tuple[
        Float[Tensor, "sample noisy_channel height width"],
        Float[Tensor, "sample clean_channel height width"],
        Float[Tensor, "sample height width"] | None,
    ]:
        view1, view2 = self.prepare_input(x, c_crossattn)
        
        # Create timestep embeddings.
        t_clean = torch.zeros_like(t)
        t1_enc = self.t_embedder_enc(t_clean)
        t1_dec = self.t_embedder_dec(t_clean)
        t2_enc = self.t_embedder_enc(t)
        t2_dec = self.t_embedder_dec(t)
        
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, t1_enc, t2_enc)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, t1_dec, t2_dec)

        # Apply output heads.
        dust3r_outputs = dict()
        with torch.cuda.amp.autocast(enabled=False):
            for subset in self.modalities_out.subsets:
                for modality, dec, im_shape in zip(
                    [subset._past_modality, subset._future_modality],
                    [dec1, dec2],
                    [shape1, shape2]
                ):
                    if modality is not None:
                        id_m = modality._id
                        dust3r_outputs[id_m] = self._downstream_head('_' + id_m, [tok.float() for tok in dec], im_shape)

        denoised, clean, weights = self.prepare_output(dust3r_outputs)
        return denoised, clean, weights
        


         
               
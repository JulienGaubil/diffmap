import sys
sys.path.append('ldm/thirdp/dust3r')

import math
import torch
import torch.nn as nn

from typing import Dict, Literal
from jaxtyping import Int, Float
from torch import Tensor
from functools import partial

from ldm.thirdp.dust3r.dust3r.model import AsymmetricCroCo3DStereo
from ldm.thirdp.dust3r.croco.models.blocks import Block, DecoderBlock
from ldm.thirdp.dust3r.dust3r.utils.misc import is_symmetrized, interleave

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


class Dust3rModel(
    AsymmetricCroCo3DStereo,
):
    def __init__(
        self,
        pos_embed: str,
        head_type: Literal["linear", "dpt"],
        depth_mode: tuple,
        conf_mode: tuple,
        img_size: list[int] = [224, 224],
        output_mode: str = "pts3d",
        enc_embed_dim: int = 1024,
        enc_depth: int = 24,
        enc_num_heads: int = 16,
        dec_embed_dim: int = 768,
        dec_depth: int = 12,
        dec_num_heads: int = 12,
        landscape_only: bool = True,
        **kwargs
    ) -> None:
        
        # Instantiate base Dust3r.
        super().__init__(
            output_mode=output_mode, head_type=head_type, depth_mode=depth_mode, conf_mode=conf_mode, landscape_only=landscape_only,
            pos_embed=pos_embed, img_size=img_size,
            enc_embed_dim=enc_embed_dim, enc_depth=enc_depth, enc_num_heads=enc_num_heads,
            dec_embed_dim=dec_embed_dim, dec_depth=dec_depth, dec_num_heads=dec_num_heads
        )

        # Replace encoding and decoding blocs with noise-conditioned blocs.
        self.enc_blocks = nn.ModuleList(
            [
                BlockTime(
                    enc_embed_dim,
                    enc_num_heads,
                    bloc,
                    mlp_ratio=kwargs.get('mlp_ratio', 4.),
                    qkv_bias=True,
                    norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                    rope=self.rope
                )
                for bloc in self.enc_blocks
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlockTime(
                    dec_embed_dim,
                    dec_num_heads,
                    bloc,
                    mlp_ratio=kwargs.get('mlp_ratio', 4.),
                    qkv_bias=True,
                    norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                    norm_mem=kwargs.get('norm_im2_in_dec', True),
                    rope=self.rope
                )
                for bloc in self.dec_blocks
            ]
        )
        self.dec_blocks2 = nn.ModuleList(
            [
                DecoderBlockTime(
                    dec_embed_dim,
                    dec_num_heads,
                    bloc,
                    mlp_ratio=kwargs.get('mlp_ratio', 4.),
                    qkv_bias=True,
                    norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)),
                    norm_mem=kwargs.get('norm_im2_in_dec', True),
                    rope=self.rope
                )
                for bloc in self.dec_blocks2
            ]
        )

        # Instantiate timestep embedder.
        self.t_embedder_enc = TimestepEmbedder(enc_embed_dim)
        self.t_embedder_dec = TimestepEmbedder(dec_embed_dim)
        nn.init.normal_(self.t_embedder_enc.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_enc.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_embedder_dec.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder_dec.mlp[2].weight, std=0.02)

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
        view1: Dict,
        view2: Dict,
        t1: Int[Tensor, "sample"],
        t2: Int[Tensor, "sample"],
        **kwargs
    ) -> tuple[
        Float[Tensor, "sample channel height width"],
        Float[Tensor, "sample channel height width"],
    ]:
        # Timestep embeddings.
        t1_enc = self.t_embedder_enc(t1)
        t1_dec = self.t_embedder_dec(t1)
        t2_enc = self.t_embedder_enc(t2)
        t2_dec = self.t_embedder_dec(t2)
        
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2, t1_enc, t2_enc)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2, t1_dec, t2_dec) # list of tokens for every decoder
        
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        return res1, res2
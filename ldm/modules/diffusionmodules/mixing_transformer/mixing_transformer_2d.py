import torch
from einops import rearrange
import torch.nn as nn


# TODO jg file definition transformer ~ U-net


class MixingTransformer2D(nn.Module):

    def __init__(self, t1, t2):
        super(MixingTransformer2D, self).__init__()
        self.t1 = t1
        self.t2 = t2

        self.config = t1.config

        self.ln1 = nn.ModuleList([
            nn.LayerNorm(normalized_shape=1152) for b in range(len(t1.transformer_blocks))
        ])

        self.ln2 = nn.ModuleList([
            nn.LayerNorm(normalized_shape=1152) for b in range(len(t1.transformer_blocks))
        ])

        self.to_qkv1 = nn.ModuleList([
            nn.Linear(in_features=1152, out_features=3*1152, bias=False) for b in range(len(t1.transformer_blocks))
        ])

        self.to_qkv2 = nn.ModuleList([
            nn.Linear(in_features=1152, out_features=3*1152, bias=False) for b in range(len(t1.transformer_blocks))
        ])

        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=1152, num_heads=16) for b in range(len(t1.transformer_blocks))
        ])

        self.fc_out1 = nn.ModuleList([
            nn.Linear(in_features=1152, out_features=1152, bias=False) for b in range(len(t1.transformer_blocks))
        ])

        self.fc_out2 = nn.ModuleList([
            nn.Linear(in_features=1152, out_features=1152, bias=False) for b in range(len(t1.transformer_blocks))
        ])

        for fc in self.fc_out1:
            torch.nn.init.zeros_(fc.weight)

        for fc in self.fc_out2:
            torch.nn.init.zeros_(fc.weight)

        # self.t1.proj_out.weight = nn.Parameter(torch.cat([self.t1.proj_out.weight, self.t1.proj_out.weight]))
        # self.t1.proj_out.bias = nn.Parameter(torch.cat([self.t1.proj_out.bias, self.t1.proj_out.bias]))
        # self.t1.out_channels *= 2
        #
        # self.t1.pos_embed.proj.weight = nn.Parameter(torch.cat([
        #     self.t1.pos_embed.proj.weight,
        #     torch.zeros_like(self.t1.pos_embed.proj.weight)
        # ], dim=1))

        # self.t1.pos_embed.proj.weight = nn.Parameter(torch.cat([self.t1.pos_embed.proj.weight]*4, dim=1))
        # self.t1.pos_embed.proj.weight = nn.Parameter(torch.cat([self.t1.pos_embed.proj.weight] + [torch.zeros_like(self.t1.pos_embed.proj.weight)] * 3, dim=1))
        self.t1.pos_embed.proj.weight = nn.Parameter(torch.cat([self.t1.pos_embed.proj.weight * .25] * 4, dim=1))  #TODO jg changer à 0.25 -> 0.5

        # self.t1.proj_out.weight = nn.Parameter(torch.cat([self.t1.proj_out.weight[:32]]*4, dim=0))
        # self.t1.proj_out.bias = nn.Parameter(torch.cat([self.t1.proj_out.bias[:32]] * 4, dim=0))

        self.t1.proj_out.weight = nn.Parameter(torch.cat([self.t1.proj_out.weight] * 4, dim=0))
        self.t1.proj_out.bias = nn.Parameter(torch.cat([self.t1.proj_out.bias] * 4, dim=0))
        # self.t1.out_channels = 16

        self.config.in_channels = 16

    def forward(
            self,
            t1_args: dict,
            t2_args: dict,
            ):

        t1_args = self.t1.pre_(**t1_args)
        t2_args = self.t2.pre_(**t2_args)

        # print(t1_args["embedded_timestep"])

        for i in range(len(self.t1.transformer_blocks)):
            t1_args["block_idx"] = i
            t2_args["block_idx"] = i

            t1_args = self.t1.single_block_(**t1_args)
            t2_args = self.t2.single_block_(**t2_args)

            t1_args, t2_args = self.mix_attention(t1_args, t2_args, i)



        t1_out = self.t1.out_(**t1_args)
        print("near end of forward")
        print(torch.allclose(t1_out[0][:, 0:8], t1_out[0][:, 8:16], atol=1e-4))
        print(torch.allclose(t1_out[0][:, 0:4], t1_out[0][:, 8:12], atol=1e-4))
        t1_out = list(t1_out)
        t1_out[0] = torch.cat([t1_out[0][:,8*i:8*i+4] for i in range(4)], dim=1)
        t1_out = tuple(t1_out)
        print("t out shape", t1_out[0].shape)
        print("after slice")
        print(torch.allclose(t1_out[0][:, 0:4], t1_out[0][:, 4:8], atol=1e-4))
        # t2_out = self.t2.out_(**t2_args)

        return t1_out

    def mix_attention(self, t1_args: dict, t2_args: dict, block_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        h1 = t1_args["hidden_states"]
        h2 = t2_args["hidden_states"]

        r1 = h1
        h1 = self.ln1[block_idx](h1)
        qkv1 = self.to_qkv1[block_idx](h1)
        q1, k1, v1 = torch.chunk(qkv1, 3, dim=-1)

        r2 = h2
        h2 = self.ln2[block_idx](h2)
        qkv2 = self.to_qkv2[block_idx](h2)
        q2, k2, v2 = torch.chunk(qkv2, 3, dim=-1)

        q = torch.cat([q1, q2], dim=-2)
        k = torch.cat([k1, k2], dim=-2)
        v = torch.cat([v1, v2], dim=-2)

        out = self.attentions[block_idx](q, k, v, need_weights=False)[0]

        out1, out2 = torch.chunk(out, 2, dim=-2)

        out1 = self.fc_out1[block_idx](out1)
        out2 = self.fc_out2[block_idx](out2)

        t1_args["hidden_states"] = out1 + r1
        t2_args["hidden_states"] = out2 + r2

        return t1_args, t2_args

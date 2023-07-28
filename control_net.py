# implementation of control net

import torch
import torch.nn as nn
import functools

from models.ddpm_entities import (
    ResnetBlockDDPM,
    Downsample,
    ddpm_conv3x3,
    get_act,
    get_timestep_embedding
)


def zero_conv(in_ch, out_ch):
    layer = nn.Conv2d(in_ch, out_ch, 1)
    nn.init.zeros_(layer.bias)
    nn.init.zeros_(layer.weight)
    return layer


class ControlNet(nn.Module):
    def __init__(self, config, ddpm):
        super().__init__()
        self.ddpm = ddpm
        self.config = config
        self.act = act = get_act(config)

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]

        from models.ddpm_entities import AttnBlock
        AttnBlock = functools.partial(AttnBlock)
        self.conditional = config.model.conditional

        modules = []
        ResnetBlock = functools.partial(ResnetBlockDDPM, act=act, temb_dim=None, dropout=dropout)

        self.centered = config.data.centered
        channels = config.data.num_channels

        # controlnet
        modules.append(nn.Embedding(config.model.num_classes, 32))
        modules.append(nn.Sequential(
            zero_conv(1, 1),
            act)
        )

        # Downsampling block
        modules.append(ddpm_conv3x3(channels, nf))
        hs_c = [nf]
        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)
            if i_level != num_resolutions - 1:
                modules.append(Downsample(channels=in_ch, with_conv=resamp_with_conv))
                hs_c.append(in_ch)

            modules.append(
                nn.Sequential(
                    zero_conv(in_ch, 64)
                )
            )

        self.all_modules = nn.ModuleList(modules)

    def downsample(self, modules, h, m_idx, num_resolutions, num_res_blocks, attn_resolutions, temb, zero_convs=None):
        hs = [modules[m_idx](h)]
        m_idx += 1
        zeros = []

        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                h = modules[m_idx](hs[-1], temb)
                m_idx += 1
                if h.shape[-1] in attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1
            if zero_convs is not None:
                zeros.append(modules[m_idx](hs[-1]))
                m_idx += 1

        return hs, m_idx, zeros

    def middle_part(self, modules, h, m_idx, temb):
        h = modules[m_idx](h, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, temb)
        m_idx += 1

        return h, m_idx

    def upsample(self, modules, h, hs, m_idx, num_resolutions, num_res_blocks, attn_resolutions, temb, zero_outputs):
        for i_level in reversed(range(num_resolutions)):
            if i_level > 0:
                h += zero_outputs[i_level - 1]

            for i_block in range(num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
                m_idx += 1
            if h.shape[-1] in attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1
        return h, m_idx

    def forward(self, x, timesteps, y):
        modules = self.all_modules
        ddpm_modules = self.ddpm.all_modules

        m_idx = 0
        m_idx_control = 0

        if self.conditional:
            # timestep/scale embedding
            temb = get_timestep_embedding(timesteps, self.nf)
            temb = ddpm_modules[m_idx](temb)
            m_idx += 1
            temb = ddpm_modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.

        h_control = h + modules[1](modules[0](y).view(-1, 1, 32, 1))

        m_idx_control += 2

        # Downsampling block from ddpm
        hs, m_idx, _ = self.downsample(ddpm_modules, h, m_idx, self.num_resolutions, self.num_res_blocks,
                                       self.attn_resolutions, temb)

        # Downsampling from controlnet
        _, m_idx_control, zero_outputs = self.downsample(modules, h_control, m_idx_control, self.num_resolutions,
                                                         self.num_res_blocks, self.attn_resolutions, None, True)

        h, m_idx = self.middle_part(ddpm_modules, hs[-1] + zero_outputs[-1], m_idx, temb)

        # Upsampling block
        h, m_idx = self.upsample(ddpm_modules, h, hs, m_idx, self.num_resolutions, self.num_res_blocks,
                                 self.attn_resolutions, temb, zero_outputs[:-1])

        assert not hs
        h = self.act(ddpm_modules[m_idx](h))
        m_idx += 1
        h = ddpm_modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(ddpm_modules)
        assert m_idx_control == len(modules)

        return h

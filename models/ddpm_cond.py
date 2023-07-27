import torch.nn as nn
import torch
import torch.nn.functional as F

import functools

from models.ddpm_entities import (
    ResnetBlockDDPM,
    Upsample,
    Downsample,
    ddpm_conv3x3,
    get_act,
    default_init,
    get_timestep_embedding,
    ResnetBlockDDPMGroup
)

class ResnetBlockDDPMConditional(ResnetBlockDDPMGroup):
    """The ResNet Blocks used in DDPM."""

    def __init__(self, act, in_ch, num_classes, class_embed_size, out_ch=None, temb_dim=None, conv_shortcut=False, dropout=0.1):
        super().__init__(act, in_ch + class_embed_size, out_ch, temb_dim, conv_shortcut, dropout)

        self.linear_map_class = nn.Sequential(
            nn.Linear(num_classes, class_embed_size)
        )
    
    def forward(self, x, c, temb=None):
        #print(x.shape)
        emb_c  = self.linear_map_class(c)
        emb_c = emb_c.view(*emb_c.shape, 1, 1)
        emb_c = emb_c.expand(-1, -1, x.shape[-2], x.shape[-1])

        x = torch.cat([x, emb_c], dim=1)
        #print(x.shape)
        return super().forward(x, temb)


class DDPMCond(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act = act = get_act(config)

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_size // (2 ** i) for i in range(num_resolutions)]
        self.num_classes = config.model.num_classes
        self.class_embed_size = config.model.class_embed_size
    
        from models.ddpm_entities import AttnBlockGroup as AttnBlock
        AttnBlock = functools.partial(AttnBlock)
        self.conditional = conditional = config.model.conditional

        modules = []
        ResnetBlock = functools.partial(ResnetBlockDDPMConditional, act=act, num_classes=self.num_classes, 
                                        class_embed_size=self.class_embed_size, temb_dim=4 * nf, dropout=dropout)
        if conditional:
            # Condition on noise levels.
            modules = [nn.Linear(nf, nf * 4)]
            modules[0].weight.data = default_init()(modules[0].weight.data.shape)
            nn.init.zeros_(modules[0].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[1].weight.data = default_init()(modules[1].weight.data.shape)
            nn.init.zeros_(modules[1].bias)

        self.centered = config.data.centered
        channels = config.data.num_channels

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

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=64))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=64))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                #print(in_ch, hs_c[-1])
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(), out_ch=out_ch))
                in_ch = out_ch
            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))
            if i_level != 0:
                modules.append(Upsample(channels=in_ch, with_conv=resamp_with_conv))

        assert not hs_c
        modules.append(nn.GroupNorm(num_channels=in_ch, num_groups=1, eps=1e-6))
        modules.append(ddpm_conv3x3(in_ch, channels, init_scale=0.))
        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, timesteps, c):
        modules = self.all_modules
        m_idx = 0
        if self.conditional:
            # timestep/scale embedding
            temb = get_timestep_embedding(timesteps, self.nf)
            temb = modules[m_idx](temb)
            m_idx += 1
            temb = modules[m_idx](self.act(temb))
            m_idx += 1
        else:
            temb = None

        if self.centered:
            # Input is in [-1, 1]
            h = x
        else:
            # Input is in [0, 1]
            h = 2 * x - 1.

        # Downsampling block
        hs = [modules[m_idx](h)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                #print(hs[-1].shape, temb.shape)
                h = modules[m_idx](hs[-1], c, temb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(modules[m_idx](hs[-1]))
                m_idx += 1

        h = hs[-1]
        h = modules[m_idx](h, c, temb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, c, temb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                #print('before')
                #print(hs[-1].shape)
                #print(h.shape)
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), c, temb)
                m_idx += 1
            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1
            if i_level != 0:
                h = modules[m_idx](h)
                m_idx += 1

        assert not hs
        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        assert m_idx == len(modules)

        return h

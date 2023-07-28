import sys
sys.path.append('..')


from configs.default_mnist_config import create_default_mnist_config
from diffusion_control_net import ControlNetRunner

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = ControlNetRunner(config, 'ddpm_checkpoints/ddpm_cont_reversed-50000.pth')

diffusion.train()
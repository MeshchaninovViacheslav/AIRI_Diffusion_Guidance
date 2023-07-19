from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = DiffusionRunner(config)

diffusion.train()

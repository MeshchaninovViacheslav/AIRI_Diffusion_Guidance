from classifierfree_mnist_config import create_default_mnist_config
from diffusion_cond import DiffusionRunnerConditional

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = DiffusionRunnerConditional(config)

diffusion.train()

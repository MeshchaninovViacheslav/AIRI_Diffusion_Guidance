from ..configs.classifierfree_mnist_config import create_default_mnist_config
from diffusion_cond import DiffusionRunnerConditional

import os

config = create_default_mnist_config()
diffusion = DiffusionRunnerConditional(config)

diffusion.train()

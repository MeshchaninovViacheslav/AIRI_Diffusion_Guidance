from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import torchvision.utils as tvu
import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = DiffusionRunner(config, eval=True)

save_path = 'mnist_generated_samples'
os.makedirs(save_path, exist_ok=True)

for i in tqdm.tqdm(range(1)):
    x_pred = diffusion.inference()
    tvu.save_image(x_pred[0], f'{save_path}/еуые.png')
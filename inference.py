from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import torchvision.utils as tvu
import tqdm
import cv2
import os
import math
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = DiffusionRunner(config, eval=True)

save_path = 'mnist_generated_samples'
os.makedirs(save_path, exist_ok=True)

i = 0
for _ in tqdm.tqdm(range(1)):
    x_pred = diffusion.inference()
    print(x_pred.shape)
    for x_p in x_pred:
        nrow = int(math.sqrt(1))
        grid = tvu.make_grid(x_p, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        cv2.imwrite(f'{save_path}/{i:05}.png', grid)
        i += 1
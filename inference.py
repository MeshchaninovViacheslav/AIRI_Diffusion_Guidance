from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import torchvision.utils as tvu
import tqdm
import cv2

import numpy as np
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = DiffusionRunner(config, eval=True)

save_path = 'mnist_generated_samples'
os.makedirs(save_path, exist_ok=True)

batch_size = 2048

i = 0
for label_idx in tqdm.tqdm(range(5)):
    x_pred = diffusion.inference(batch_size)
    for j, x_p in enumerate(x_pred):
        cv2.imwrite(f'{save_path}/{i}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
        i += 1 

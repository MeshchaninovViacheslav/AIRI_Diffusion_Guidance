from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import torchvision.utils as tvu
import tqdm
import torch
import cv2
import numpy as np


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

config = create_default_mnist_config()
diffusion = DiffusionRunner(config, eval=True)

save_path = 'mnist_classifier_guidance_generated_samples'
os.makedirs(save_path, exist_ok=True)

batch_size = 5

i = 0
for _ in tqdm.tqdm(range(1)):
    labels = torch.randint(low=0, high=10, size=(batch_size,)).to(config.device)
    x_pred = diffusion.inference(batch_size, labels)
    for j, x_p in enumerate(x_pred):
        cv2.imwrite(f'{save_path}/{i}_class_{labels[j]}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
        #tvu.save_image(x_p, f'{save_path}/{i}_class_{labels[j]}.png')
        i += 1
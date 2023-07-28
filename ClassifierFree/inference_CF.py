import sys
sys.path.append('../configs')
sys.path.append('..')


from configs.classifierfree_mnist_config import create_default_mnist_config
from diffusion_cond import DiffusionRunnerConditional

import torchvision.utils as tvu
import tqdm
import torch
import cv2
import numpy as np
import math

import os

config = create_default_mnist_config()
diffusion = DiffusionRunnerConditional(config, eval=True)

save_path = '../generations/mnist_classifier_free_example_grid'
os.makedirs(save_path, exist_ok=True)

batch_size = 100

labels = np.tile(np.arange(10), (batch_size, 1)).T
labels = torch.Tensor(labels).to(config.device).long()
i = 0
for label_idx, label in tqdm.tqdm(enumerate(labels)):
    x_pred = diffusion.inference(batch_size, label)
    # for j, x_p in enumerate(x_pred):
    #     cv2.imwrite(f'{save_path}/{i}_class_{label_idx}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
    #     i += 1 
    nrow = int(math.sqrt(len(label)))
    grid = tvu.make_grid(x_pred, nrow=nrow).permute(1, 2, 0)
    grid = grid.data.numpy().astype(np.uint8)
    cv2.imwrite(f'{save_path}/{label_idx}.png', grid)
import sys
sys.path.append('../configs')
sys.path.append('..')

from configs.default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner
import tqdm
import torch
import cv2
import numpy as np
import torchvision.utils as tvu

import math




import os
from diffusion_control_net import ControlNetRunner

config = create_default_mnist_config()
diffusion = ControlNetRunner(config, '../ddpm_checkpoints/ddpm_cont_reversed-50000.pth', eval=True)

save_path = '../generations/mnist_control_net_examples'
os.makedirs(save_path, exist_ok=True)

batch_size = 100

i = 0
labels = np.tile(np.array([3, 5]), (batch_size, 1)).T
labels = torch.Tensor(labels).to(config.device).long()
# for label_idx, label in tqdm.tqdm(enumerate(labels)):
#     x_pred = diffusion.inference(batch_size, label)
#     for j, x_p in enumerate(x_pred):
#         cv2.imwrite(f'{save_path}/{i}_class_{label_idx}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
#         i += 1 
for label_idx, label in tqdm.tqdm(enumerate(labels)):
    x_pred = diffusion.inference(batch_size, label)
    # for j, x_p in enumerate(x_pred):
    #     cv2.imwrite(f'{save_path}/{i}_class_{label_idx}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
    #     i += 1 
    nrow = int(math.sqrt(len(label)))
    grid = tvu.make_grid(x_pred, nrow=nrow).permute(1, 2, 0)
    grid = grid.data.numpy().astype(np.uint8)
    cv2.imwrite(f'{save_path}/{label_idx}.png', grid)
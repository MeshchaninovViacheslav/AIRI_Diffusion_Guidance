from default_mnist_config import create_default_mnist_config
from diffusion import DiffusionRunner

import torchvision.utils as tvu
import tqdm
import torch
import cv2
import numpy as np
import math
import torchvision
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


config = create_default_mnist_config()
diffusion = DiffusionRunner(config, eval=True)

save_path = 'mnist_classifier_guidance_set'
os.makedirs(save_path, exist_ok=True)

batch_size = 1000

i = 0
# for _ in tqdm.tqdm(range(1)):
#     labels = torch.randint(low=0, high=10, size=(batch_size,)).to(config.device)
#     x_pred = diffusion.inference(batch_size, labels)

#     nrow = int(math.ceil(math.sqrt(batch_size)))
#     grid = torchvision.utils.make_grid(x_pred, nrow=nrow).permute(1, 2, 0)
#     grid = grid.data.numpy().astype(np.uint8)
#     cv2.imwrite(f'{save_path}/{i}.png', grid)
#    # for j, x_p in enumerate(x_pred):
#     #    cv2.imwrite(f'{save_path}/{i}_class_{labels[j]}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
#         #tvu.save_image(x_p, f'{save_path}/{i}_class_{labels[j]}.png')
#     i += 1
# labels = np.tile(np.arange(10), (batch_size, 1)).T
# labels = torch.Tensor(labels).to(config.device).long()
# for i, label in tqdm.tqdm(enumerate(labels)):
#     x_pred = diffusion.inference(batch_size, label)
    
#     nrow = int(math.ceil(math.sqrt(batch_size)))
#     grid = torchvision.utils.make_grid(x_pred, nrow=nrow).permute(1, 2, 0)
#     grid = grid.data.numpy().astype(np.uint8)
#     cv2.imwrite(f'{save_path}/{i}_{config.classifier.gamma}.png', grid)

labels = np.tile(np.arange(10), (batch_size, 1)).T
labels = torch.Tensor(labels).to(config.device).long()
i = 0
for label_idx, label in tqdm.tqdm(enumerate(labels)):
    x_pred = diffusion.inference(batch_size, label)
    for j, x_p in enumerate(x_pred):
        cv2.imwrite(f'{save_path}/{i}_class_{label_idx}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
        i += 1 
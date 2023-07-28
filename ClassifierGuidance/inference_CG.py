from ..configs.default_mnist_config import create_default_mnist_config
from ..diffusion import DiffusionRunner
import tqdm
import torch
import cv2
import numpy as np




import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'


config = create_default_mnist_config()
diffusion = DiffusionRunner(config, eval=True)

save_path = '..generations/mnist_classifier_guidance_set'
os.makedirs(save_path, exist_ok=True)

batch_size = 1000

i = 0
labels = np.tile(np.arange(10), (batch_size, 1)).T
labels = torch.Tensor(labels).to(config.device).long()
for label_idx, label in tqdm.tqdm(enumerate(labels)):
    x_pred = diffusion.inference(batch_size, label)
    for j, x_p in enumerate(x_pred):
        cv2.imwrite(f'{save_path}/{i}_class_{label_idx}.png', x_p.permute(1,2,0).numpy().astype(np.uint8))
        i += 1 
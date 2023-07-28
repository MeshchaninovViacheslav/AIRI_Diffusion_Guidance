import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
import sys
sys.append.path('..')

@torch.no_grad()
def accuracy(output, target):
    """Computes the accuracy"""
    assert output.shape == target.shape
    return torch.sum(output == target) / target.size(0)

def compute_fid(reals, fakes):
    metric = FrechetInceptionDistance(feature=768, normalize=True).to('cuda')
    
    metric.update(reals, real=True)
    metric.update(fakes, real=False)
    
    return metric.compute()

metric = FrechetInceptionDistance(feature=768, normalize=True).to('cuda')

if __name__ == "__main__":
    import os
    import cv2
    from configs.default_mnist_config import create_default_mnist_config
    from data_generator import DataGenerator
    import numpy as np
    
    config = create_default_mnist_config()
    
    dataloader = DataGenerator(config)
    
    real_images = torch.tensor([])
    
    for (x, y) in tqdm(dataloader.valid_loader):
        x = torch.cat((x, x, x), dim=1) / 255.
        #real_images = torch.cat((real_images, x), dim=0)
        #print(x.size())
        metric.update(x.to('cuda'), real=True)
    
    #real_images = real_images.to('cuda')
    #real_images = real_images[:2000]
    print(real_images.shape)
    
    fake_path = 'mnist_classifier_free_set'
    fake_list = os.listdir(fake_path)
    
    fake_images = []
    
    for i, image in tqdm(enumerate(fake_list)):
        if '.png' not in image: continue
        img = cv2.imread(f'{fake_path}/{image}')
        img = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img)[None, ...] / 255.
        fake_images.append(img_tensor)
        if i % 100 == 0:

            metric.update(torch.cat(fake_images).to('cuda'), real=False)
            fake_images = []
    
    fake_images = torch.cat(fake_images)
    print(fake_images.shape)
    fake_images = fake_images.to('cuda')
    #result = compute_fid(real_images, fake_images)
    result = metric.compute()

    print(f'FID result: {result}')
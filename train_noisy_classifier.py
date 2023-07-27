import torch
import wandb

from default_mnist_config import create_default_mnist_config
from models.classifier import ResNet, ResidualBlock
from data_generator import DataGenerator
from tqdm.auto import trange
from tqdm import tqdm
import os
from ddpm_sde import DDPM_SDE

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def log_metric(metric_name, loader_name, value, step):
    wandb.log({f'{metric_name}/{loader_name}': value}, step=step)


def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape == target.shape
        return torch.sum(pred == target) / target.size(0)

wandb.init(project='sde', name='noisy_classifier')

device = torch.device('cuda')
classifier_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}
model = ResNet(**classifier_args)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

config = create_default_mnist_config()
datagen = DataGenerator(config)
train_generator = datagen.sample_train()
val_generator = datagen.valid_loader

#SDE
sde = DDPM_SDE(config=config)

TOTAL_ITERS = 2_000
EVAL_FREQ = 200

eps = 1e-5

model.train()

for iter_idx in trange(1, 1 + TOTAL_ITERS):

    """
    train
    """
    (X, y) = next(train_generator)

    #sample timestep t
    t = torch.FloatTensor(X.shape[0]).uniform_() * (1 - eps) + eps 
    # create noise
    noise = torch.randn_like(X)
    mean, std = sde.marginal_prob(X, t)
    std = std.view(-1, 1, 1, 1)
    # inject noise
    X = mean + noise * std

    X = X.to(device)
    y = y.to(device)
    output = model(X)
    optimizer.zero_grad()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
   # print(loss.item())
    log_metric('loss', 'train', loss.item(), step=iter_idx)

    if iter_idx % EVAL_FREQ == 0:
        with torch.no_grad():
            model.eval()
            
            size = 5 * config.training.batch_size
            #T = tqdm(enumerate(val_generator))
            small_noise_t = torch.ones(size) * eps
            little_noise_t = (torch.ones(size)) / 5
            middle_noise_t = (torch.ones(size))  / 2 
            hard_noise_t = (torch.ones(size)) - 1 / 4 
            noises = {'small': small_noise_t,
                      'little': little_noise_t,
                      'middle': middle_noise_t,
                      'hard': hard_noise_t}
            for name, noise_t in tqdm(noises.items()):
                valid_loss = 0
                valid_accuracy = 0
                T = enumerate(val_generator)
                for i, (X, y) in T:
                    #sample timestep t
                    
                    # create noise
                    noise = torch.randn_like(X)
                 #   print(X.size(), noise_t.size())
                    mean, std = sde.marginal_prob(X, noise_t)
                    std = std.view(-1, 1, 1, 1)
                    # inject noise
                    X = mean + noise * std
                    
                    X = X.to(device)
                    y = y.to(device)

                    output = model(X)
                    loss = criterion(output, y)
                    valid_loss += loss.item()

                    acc = accuracy(output, y)
                    valid_accuracy += acc.item()

                log_metric('cross_entropy', 'valid_{}'.format(name), valid_loss / len(val_generator), step=iter_idx)
                log_metric('accuracy', 'valid_{}'.format(name), valid_accuracy / len(val_generator), step=iter_idx)
        model.train()
model.eval()

torch.save(model.state_dict(), './ddpm_checkpoints/noisy_classifier.pth')

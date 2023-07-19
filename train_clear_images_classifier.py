import torch
import wandb

from default_mnist_config import create_default_mnist_config
from models.classifier import ResNet, ResidualBlock
from data_generator import DataGenerator
from tqdm.auto import trange
from tqdm import tqdm
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def log_metric(metric_name, loader_name, value, step):
    wandb.log({f'{metric_name}/{loader_name}': value}, step=step)


def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape == target.shape
        return torch.sum(pred == target) / target.size(0)

wandb.init(project='sde', name='clean_classifier')

device = torch.device('cuda')
classifier_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}
model = ResNet(**classifier_args)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

datagen = DataGenerator(create_default_mnist_config())
train_generator = datagen.sample_train()
val_generator = datagen.valid_loader

TOTAL_ITERS = 2_000
EVAL_FREQ = 500

model.train()

for iter_idx in trange(1, 1 + TOTAL_ITERS):

    """
    train
    """
    (X, y) = next(train_generator)
    X = X.to(device)
    y = y.to(device)
    output = model(X)
    optimizer.zero_grad()
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    log_metric('loss', 'train', loss.item(), step=iter_idx)

    if iter_idx % EVAL_FREQ == 0:
        with torch.no_grad():
            model.eval()
            valid_loss = 0
            valid_accuracy = 0

            T = tqdm(enumerate(val_generator))
            for i, (X, y) in T:
                X = X.to(device)
                y = y.to(device)

                output = model(X)
                loss = criterion(output, y)
                valid_loss += loss.item()

                acc = accuracy(output, y)
                valid_accuracy += acc.item()

            log_metric('cross_entropy', 'valid', valid_loss / len(val_generator), step=iter_idx)
            log_metric('accuracy', 'valid', valid_accuracy / len(val_generator), step=iter_idx)
        model.train()
model.eval()

torch.save(model.state_dict(), './ddpm_checkpoints/clean_classifier.pth')

import torch
import wandb

from ..configs.default_mnist_config import create_default_mnist_config
from ..models.classifier import ResNet, ResidualBlock
from ..data_generator import CustomDataGenerator
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix


def log_metric(metric_name, loader_name, value, step):
    wandb.log({f'{metric_name}/{loader_name}': value}, step=step)


def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape == target.shape
        return torch.sum(pred == target) / target.size(0)

#wandb.init(project='sde', name='noisy_classifier')

save_path = '..generations/mnist_classifier_free_set/'
model_load_path = '..ddpm_checkpoints/clean_classifier.pth'

device = torch.device('cuda')
classifier_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
}
model = ResNet(**classifier_args)
classifier_checkpoint = torch.load(model_load_path, map_location='cpu')
model.load_state_dict(classifier_checkpoint)
model.to(device)
model.eval()


config = create_default_mnist_config()
dataset = CustomDataGenerator(config, save_path)
loader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False, drop_last=False)

golds = []
preds = []
valid_accuracy = 0
for (X,y) in tqdm(loader):
    X = X.to(device)
    y = y.to(device)
    output = model(X)
    acc = accuracy(output, y)
    valid_accuracy += acc.item()

    with torch.no_grad():
        pred = torch.argmax(output, dim=1).detach().cpu().tolist()

    preds.extend(pred)
    golds.extend(y.detach().cpu().tolist())


print(valid_accuracy / len(loader))
confusion_matrix(golds, preds)
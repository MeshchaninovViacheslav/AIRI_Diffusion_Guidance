import torch
import torchvision
import wandb
import os
import math

import numpy as np

from models.ddpm import DDPM
from models.ema import ExponentialMovingAverage
from ddpm_sde import DDPM_SDE, EulerDiffEqSolver
from data_generator import DataGenerator

from ml_collections import ConfigDict
from typing import Optional, Union
from tqdm.auto import trange
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        self.config = config
        self.eval = eval

        self.model = DDPM(config=config)
        self.sde = DDPM_SDE(config=config)
        self.diff_eq_solver = EulerDiffEqSolver(
            self.sde,
            self.calc_score,
            ode_sampling=config.training.ode_sampling
        )
        self.inverse_scaler = lambda x: torch.clip(127.5 * (x + 1), 0, 255)

        self.checkpoints_folder = config.training.checkpoints_folder
        if eval:
            self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            self.restore_parameters()
            self.switch_to_ema()

        device = torch.device(self.config.device)
        self.device = device
        self.model.to(device)

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        #AIRI_Diffusion_Guidance/ddpm_checkpoints/ddpm_cont_newt-1.pth
        #AIRI_Diffusion_Guidance/ddpm_checkpoints/ddpm_cont-50000.pth
        #AIRI_Diffusion_Guidance/ddpm_checkpoints/ddpm_cont_reversed-50000.pth
        print(checkpoints_folder + self.config.chkp_name)
        model_ckpt = torch.load(checkpoints_folder + self.config.chkp_name, map_location=device)['model']
        self.model.load_state_dict(model_ckpt)

        ema_ckpt = torch.load(checkpoints_folder + self.config.chkp_name, map_location=device)['ema']
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.grad_scaler = GradScaler()
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.model.parameters()]))

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )

        self.log_metric('grad_norm', 'train', grad_norm.item())
        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        self.ema.update(self.model.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.sde.T - eps) + eps

    def calc_score(self, input_x: torch.Tensor, input_t: torch.Tensor, y=None) -> torch.Tensor:
        """
        calculate score w.r.t noisy X and t
        input:
            input_x - noizy image
            input_t - time label
        algorithm:
            1) predict noize via DDPM
            2) calculate std of input_x
            3) calculate score = -pred_noize / std
        """
        eps = self.model(input_x, input_t)
        std = self.sde.marginal_std(input_t)
        std = std.view(-1, 1, 1, 1)
        score = -eps / std
        return {
            'score': score,
            'noise': eps
        }

    def calc_loss(self, clean_x: torch.Tensor, eps: float = 1e-5) -> Union[float, torch.Tensor]:
        """
        Define score-matching MSE loss
        input:
            clean_x - clean image which is fed to network
        output:

        algorithm:
            1) sample time - t
            2) find conditional distribution q(x_t | x_0), x_0 = clean_x
            3) sample x_t ~ q(x_t | x_0), x_t = noisy_x
            4) calculate predicted score via self.calc_score
            5) true score = -z / std
            6) loss = mean(torch.pow(score + pred_score, 2))
        """
        t = self.sample_time(clean_x.shape[0], eps)
        mean, std = self.sde.marginal_prob(clean_x, t)
        noise = torch.randn_like(clean_x)
        std = std.view(-1, 1, 1, 1)
        pred = self.calc_score(mean + noise * std, t)
        score = -noise / self.sde.marginal_std(t).view(-1, 1, 1, 1)
        loss = torch.pow(pred['noise'] - noise, 2).mean()
        return loss

    def set_data_generator(self) -> None:
        self.datagen = DataGenerator(self.config)

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def train(self) -> None:
        self.set_optimizer()
        self.set_data_generator()
        train_generator = self.datagen.sample_train()
        self.step = 0

        wandb.init(project='sde', name=self.config.training.exp_name)

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        self.model.train()
        for iter_idx in trange(1, 1 + self.config.training.training_iters):
            self.step = iter_idx

            (X, y) = next(train_generator)
            X = X.to(self.device)
            with torch.cuda.amp.autocast():
                loss = self.calc_loss(clean_x=X)
            
            if iter_idx % self.config.training.logging_freq == 0:
                self.log_metric('loss', 'train', loss.item())
            
            self.optimizer_step(loss)

            if iter_idx % self.config.training.snapshot_freq == 0:
                self.snapshot()

            if iter_idx % self.config.training.eval_freq == 0:
                self.validate()

            if iter_idx % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

        self.model.eval()
        self.save_checkpoint()
        self.switch_to_ema()

    def validate(self) -> None:
        prev_mode = self.model.training

        self.model.eval()
        self.switch_to_ema()

        valid_loss = 0
        valid_count = 0
        with torch.no_grad():
            for (X, y) in self.datagen.valid_loader:
                X = X.to(self.device)
                loss = self.calc_loss(clean_x=X)
                valid_loss += loss.item() * X.size(0)
                valid_count += X.size(0)

        valid_loss = valid_loss / valid_count
        self.log_metric('loss', 'valid_loader', valid_loss)

        self.switch_back_from_ema()
        self.model.train(prev_mode)

    def save_checkpoint(self) -> None:
        os.makedirs(self.checkpoints_folder, exist_ok=True)
        prefix = f"{self.config.checkpoints_prefix}-{self.step}"
        file_path = os.path.join(self.checkpoints_folder, prefix + ".pth")
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.grad_scaler.state_dict(),
                "step": self.step,
            },
            file_path
        )
        print(f"Save model to: {file_path}")

    def sample_images(
            self, batch_size: int,
            eps: float = 1e-5,
            labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.num_channels,
            self.config.data.image_size,
            self.config.data.image_size
        )
        device = torch.device(self.config.device)
        with torch.no_grad():
            """
            Implement cycle for Euler RSDE sampling w.r.t labels  
            """
            noisy_x = torch.randn(shape, device=device)
            times = torch.linspace(self.sde.T - eps, 0, self.sde.N, device=device) + eps
            for time in times:
                t = torch.ones(batch_size, device=device) * time
                noisy_x, _ = self.diff_eq_solver.step(noisy_x, t)

        return self.inverse_scaler(noisy_x)

    def snapshot(self, labels: Optional[torch.Tensor] = None) -> None:
        prev_mode = self.model.training

        self.model.eval()
        self.switch_to_ema()

        images = self.sample_images(self.config.training.snapshot_batch_size, labels=labels).cpu()
        nrow = int(math.sqrt(self.config.training.snapshot_batch_size))
        grid = torchvision.utils.make_grid(images, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        self.log_metric('images', 'from_noise', wandb.Image(grid))

        self.switch_back_from_ema()
        self.model.train(prev_mode)
        
    def inference(self, batch_size, labels = None) -> torch.Tensor:
        self.model.eval()
        self.switch_to_ema()

        images = self.sample_images(batch_size, labels=labels).cpu()
        images = images.type(torch.uint8)
        self.switch_back_from_ema()
        return images
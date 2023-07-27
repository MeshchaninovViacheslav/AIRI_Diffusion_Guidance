import torch
import torchvision
import wandb
import os
import math

import numpy as np

from models.ddpm_cond import DDPMCond
from models.ema import ExponentialMovingAverage
from ddpm_sde_cond import DDPM_SDECond as DDPM_SDE, EulerDiffEqSolverCond as EulerDiffEqSolver
from data_generator import DataGenerator
from torch.nn.functional import one_hot

from ml_collections import ConfigDict
from typing import Optional, Union
from tqdm.auto import trange
from timm.scheduler.cosine_lr import CosineLRScheduler
from torch.cuda.amp import GradScaler
from diffusion import DiffusionRunner

class DiffusionRunnerConditional(DiffusionRunner):
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        super().__init__(config, eval)
        self.config = config

        self.model = DDPMCond(config=config)
        self.sde = DDPM_SDE(config=config)
        self.diff_eq_solver = EulerDiffEqSolver(
            self.sde,
            self.calc_score_classifier_free,
            ode_sampling=config.training.ode_sampling
        )
        
        device = torch.device(self.config.device)
        self.device = device
        self.model.to(device)

    def calc_score(self, input_x: torch.Tensor, input_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
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
        cond = one_hot(cond, self.config.model.num_classes).float()
        is_class_cond = torch.rand(size=(input_x.shape[0],1), device=input_x.device) >= self.config.training.p_uncond
        cond = cond * is_class_cond
        
        eps = self.model(input_x, input_t, cond)
        std = self.sde.marginal_std(input_t)
        std = std.view(-1, 1, 1, 1)
        score = -eps / std
        return {
            'score': score,
            'noise': eps
        }
    
    def calc_score_classifier_free(self, input_x: torch.Tensor, input_t: torch.Tensor, cond: torch.Tensor):
        cond = one_hot(cond, self.config.model.num_classes).float()
        
        #print(input_x.device, input_t.device, cond.device)
        
        eps1 = self.model(input_x, input_t, cond)
        eps2 = self.model(input_x, input_t, cond * 0)
        eps = (1 + 0.1) * eps1 - 0.1 * eps2
        std = self.sde.marginal_std(input_t)
        std = std.view(-1, 1, 1, 1)
        score = -eps / std
        
        return {
            'score': score,
            'noise': eps
        }

    def calc_loss(self, clean_x: torch.Tensor, cond: torch.Tensor, eps: float = 1e-5) -> Union[float, torch.Tensor]:
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
        pred = self.calc_score(mean + noise * std, t, cond)
        score = -noise / self.sde.marginal_std(t).view(-1, 1, 1, 1)
        
        loss = torch.pow(pred['noise'] - noise, 2).mean()
        
        return loss

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
            y = y.to(self.device)
            
            with torch.cuda.amp.autocast():
                loss = self.calc_loss(clean_x=X, cond=y)
            
            if iter_idx % self.config.training.logging_freq == 0:
                self.log_metric('loss', 'train', loss.item())
            
            self.optimizer_step(loss)

            if iter_idx % self.config.training.snapshot_freq == 0:
                self.snapshot(labels=y)

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
                y = y.to(self.device)
                loss = self.calc_loss(clean_x=X, cond=y)
                valid_loss += loss.item() * X.size(0)
                valid_count += X.size(0)

        valid_loss = valid_loss / valid_count
        self.log_metric('loss', 'valid_loader', valid_loss)

        self.switch_back_from_ema()
        self.model.train(prev_mode)

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
                noisy_x, _ = self.diff_eq_solver.step(noisy_x, labels, t)

        return self.inverse_scaler(noisy_x)

    def snapshot(self, labels: Optional[torch.Tensor] = None) -> None:
        prev_mode = self.model.training

        self.model.eval()
        self.switch_to_ema()

        images = self.sample_images(len(labels), labels=labels).cpu()
        nrow = int(math.sqrt(len(labels)))
        grid = torchvision.utils.make_grid(images, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        self.log_metric('images', 'from_noise', wandb.Image(grid))

        self.switch_back_from_ema()
        self.model.train(prev_mode)
        
    def inference(self, labels = None) -> None:
        self.model.eval()
        self.switch_to_ema()

        images = self.sample_images(len(labels), labels=labels).cpu()
        self.switch_back_from_ema()
        return images
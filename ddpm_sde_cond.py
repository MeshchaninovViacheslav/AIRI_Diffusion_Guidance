import torch
import numpy as np


class DDPM_SDE:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = config.sde.N
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max
        self.predict = config.predict
        
    @property
    def T(self):
        return 1

    def sde(self, x, t):
        """
        Calculate drift coeff. and diffusion coeff. in forward SDE
        """
        drift = (-1) / 2 * self._beta(t)[:, None, None, None] * x
        diffusion = torch.sqrt(self._beta(t))
        return drift, diffusion

    def _beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def _integrate_beta(self, t):
        return (1 / 2) * (self.beta_1 - self.beta_0) * t ** 2 + self.beta_0 * t

    def marginal_prob(self, x_0, t):
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        integrated_beta = self._integrate_beta(t)
        mean = x_0 * torch.exp((-1) / 2 * integrated_beta)[:, None, None, None]
        std = torch.sqrt(1 - torch.exp(-integrated_beta))
        return mean, std

    def marginal_std(self, t):
        """
        Calculate marginal q(x_t|x_0)'s std
        """
        std = torch.sqrt(1 - torch.exp(-self._integrate_beta(t)))
        return std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def reverse(self, score_fn, ode_sampling=False):
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          ode_sampling: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_fn = self.sde
        beta_fn = self._beta
        predict = self.predict
        
        # Build the class for reverse-time SDE.
        class RSDE:
            def __init__(self):
                self.N = N
                self.ode_sampling = ode_sampling

            @property
            def T(self):
                return T

            def sde(self, x, cond, t):
                if ode_sampling:
                    drift_sde, _ = sde_fn(x, t)
                    drift = drift_sde - (1 / 2) * beta_fn(t)[:, None, None, None] * score_fn(x, t, cond)['score']
                    diffusion = 0
                else:
                    drift_sde, diffuson_sde = sde_fn(x, t)
                    drift = drift_sde - beta_fn(t)[:, None, None, None] * score_fn(x, t, cond)['score']
                    diffusion = diffuson_sde
                return drift, diffusion

        return RSDE()


class EulerDiffEqSolver:
    def __init__(self, sde, score_fn, ode_sampling=False):
        self.sde = sde
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling
        self.rsde = sde.reverse(score_fn, ode_sampling)

    def step(self, x_t, cond, t):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = -1 / self.rsde.N
        noise = torch.randn_like(x_t)
        drift, diffusion = self.rsde.sde(x_t, cond, t)
        x_mean = x_t + drift * dt
        #print(x_mean.shape, diffusion.shape, noize.shape)
        x = x_mean + diffusion.view(-1, 1, 1, 1) * np.sqrt(-dt) * noise
        return x, x_mean

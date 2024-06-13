# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

# since this code will be only used for running cortexbench
# the following dependency won't be added to the project by default
from mjrl.policies.gaussian_mlp import BatchNormMLP
from numpy.typing import NDArray


class ConvBatchNormMLP(BatchNormMLP):
    """Convolution followed with a BatchNormMLP (BatchNormMLP is from mjrl).

    Attrs:
        embedding_dim (tuple[int, ...] | list[int, ...] | torch.Size): dimension of the representation.
        proprio_dim (tuple[int, ...] | list[int, ...] | torch.Size):
            dimension of the proprio information from the environment.
        history_window (int): the number of history observations considered.
        model (nn.ModuleDict): the dict to original BatchNormMLP (as a "head") and newly created Conv (as a "neck").
        device (str | torch.device): track the device that the model is on.
    """

    def __init__(
        self,
        env_spec: Any,
        hidden_sizes: str = "(64, 64)",  # str is to adapt with mjrl side
        min_log_std: float = -3.0,
        init_log_std: float = 0.0,
        seed: Optional[int] = None,
        nonlinearity: str = "relu",
        dropout: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            env_spec (gym.EnvSpec): specs of the environment that this policy will run on.
            hidden_sizes (tuple): size of hidden layers of MLP. Defaults to (64,64).
            min_log_std (float): minimum log std value for action. This is to match mjrl. Defaults to -3.
            init_log_std (float): initial log std value for action. This is to match mjrl. Defaults to 0.
            seed (Optional[int]): seed. Defaults to None.
            nonlinearity (str): kind of non-linearility activation function. Defaults to 'relu'.
            dropout (float): dropout rate. Defaults to 0.
        """
        self.embedding_dim = kwargs["embedding_dim"]  # [C, H, W]
        self.proprio_dim = kwargs["proprio_dim"]
        self.history_window = kwargs["history_window"]
        hidden_sizes = eval(hidden_sizes)  # hack to match mjrl
        env_spec.observation_dim = hidden_sizes[0] + self.proprio_dim
        super().__init__(
            env_spec, hidden_sizes, min_log_std, init_log_std, seed, nonlinearity, dropout, *args, **kwargs
        )

        neck = nn.Sequential(
            nn.Conv2d(self.embedding_dim[0] * self.history_window, 256, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([256, 7, 7]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 14x14 -> 7x7  # just to keep the same as super class
            nn.Conv2d(256, 256, kernel_size=3, stride=2),
            nn.LayerNorm([256, 3, 3]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 7x7 -> 3x3
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.LayerNorm([256, 1, 1]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 3x3 -> 1x1
            nn.Flatten(),
        )

        # re-encapsule so that all nn parts are in self.model
        # so that explicit operations on self.model by cortexbench are applied on all nn parts
        # e.g. policy.model.eval()
        head: nn.Module = self.model  # type:ignore [has-type]
        self.model = nn.ModuleDict({"neck": neck, "head": head})
        self.device: Optional[str | torch.device] = None

    def to(self, device: str | torch.device) -> None:
        """Put the model on the `device`.

        Args:
            device (str | torch.device): the device to put the model
        """
        for k in self.model:
            self.model[k].to(device)
        self.device = device

    def eval(self) -> None:
        """Set the model in eval mode."""
        for k in self.model:
            self.model[k].eval()

    def train(self) -> None:
        """:Set the model in train mode."""
        for k in self.model:
            self.model[k].train()

    def get_action_mean(self, observation: torch.Tensor) -> torch.Tensor:
        """Get the mean action given the observation.

        Args:
            observation (torch.Tensor): observation.

        Returns:
            torch.Tensor : mean action.
        """
        if len(self.embedding_dim) > 0:
            # observation (B, T*H*W*C+C_pripro)
            if self.proprio_dim > 0:
                emb_obs, proprio_obs = observation[..., : -self.proprio_dim], observation[..., -self.proprio_dim :]
                emb_obs = rearrange(
                    emb_obs,
                    "b (t h w c) -> b (c t) h w",
                    t=self.history_window,
                    c=self.embedding_dim[0],
                    h=self.embedding_dim[1],
                    w=self.embedding_dim[2],
                )
                emb_obs = self.model["neck"](emb_obs)
                self.obs_var = torch.cat([emb_obs, proprio_obs], dim=1)
            else:
                emb_obs = rearrange(
                    observation,
                    "b (t h w c) -> b (c t) h w",
                    t=self.history_window,
                    c=self.embedding_dim[0],
                    h=self.embedding_dim[1],
                    w=self.embedding_dim[2],
                )
                self.obs_var = self.model["neck"](emb_obs)
        else:
            raise ValueError(f"input observation {observation.size()} is not from a valid spatial embedding.")
        mean = self.model["head"](self.obs_var)
        return mean

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Model forward. Wrapper for get_action_mean() used during training.

        Args:
            observation (torch.Tensor): observation.

        Returns:
            torch.Tensor: mean action.
        """
        return self.get_action_mean(observation)

    def get_action(self, observation: NDArray) -> tuple[NDArray, dict[str, Any]]:
        """Get action with some noise used in evaluation / rollout. No gradient.

        Args:
            observation (NDArray): observation.

        Returns:
            tuple[NDArray, dict[str, Any]]: action and some statistics (required by mjrl)
        """
        with torch.no_grad():
            observation = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(self.device)
            mean = self.get_action_mean(observation).detach().cpu().numpy().ravel()
            noise = np.exp(self.log_std_val) * np.random.randn(self.m)
            action = mean + noise
            return (action, {"mean": mean, "log_std": self.log_std_val, "evaluation": mean})

    def get_action_deterministic(self, observation: NDArray) -> tuple[NDArray, dict[str, Any]]:
        """Get action without noise (using mean) used in evaluation / rollout. No gradient.

        Args:
            observation (NDArray): observation.

        Returns:
            tuple[NDArray, dict[str, Any]]: action and some statistics (required by mjrl)
        """
        with torch.no_grad():
            observation = torch.from_numpy(observation.astype(np.float32)).unsqueeze(0).to(self.device)
            action = self.get_action_mean(observation).detach().cpu().numpy().ravel()
            return (action, {"mean": action, "log_std": 0, "evaluation": action})


class ConvPolicyHead(ConvBatchNormMLP):
    """A smaller Convolution followed with a smaller BatchNormMLP (BatchNormMLP is from mjrl).

    Attrs:
        embedding_dim (tuple[int, ...] | list[int, ...] | torch.Size): dimension of the representation.
        proprio_dim (tuple[int, ...] | list[int, ...] | torch.Size):
            dimension of the proprio information from the environment.
        history_window (int): the number of history observations considered.
        model (nn.ModuleDict): the dict to original BatchNormMLP (as a "head") and newly created Conv (as a "neck").
        device (str | torch.device): track the device that the model is on.
    """

    def __init__(
        self,
        env_spec: Any,
        hidden_sizes: str = "(64, 64)",  # str is to adapt with mjrl side
        min_log_std: float = -3.0,
        init_log_std: float = 0.0,
        seed: Optional[int] = None,
        nonlinearity: str = "relu",
        dropout: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Args:
            env_spec (gym.EnvSpec): specs of the environment that this policy will run on.
            hidden_sizes (tuple): size of hidden layers of MLP. Defaults to (64,64).
            min_log_std (float): minimum log std value for action. This is to match mjrl. Defaults to -3.
            init_log_std (float): initial log std value for action. This is to match mjrl. Defaults to 0.
            seed (Optional[int]): seed. Defaults to None.
            nonlinearity (str): kind of non-linearility activation function. Defaults to 'relu'.
            dropout (float): dropout rate. Defaults to 0.
        """
        self.embedding_dim = kwargs["embedding_dim"]  # [C, H, W]
        self.proprio_dim = kwargs["proprio_dim"]
        self.history_window = kwargs["history_window"]
        hidden_sizes = eval(hidden_sizes)  # hack to match mjrl
        env_spec.observation_dim = hidden_sizes[0] + self.proprio_dim
        super().__init__(
            env_spec, hidden_sizes, min_log_std, init_log_std, seed, nonlinearity, dropout, *args, **kwargs
        )

        del self.model

        neck = nn.Sequential(
            nn.Conv2d(self.embedding_dim[0] * self.history_window, 60, kernel_size=4, stride=2, padding=1),
            nn.LayerNorm([60, 7, 7]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 14x14-> 7x7  # just to keep the same as super class
            nn.Conv2d(60, 60, kernel_size=3, stride=2),
            nn.LayerNorm([60, 3, 3]),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),  # 3x3
            nn.Flatten(),
        )
        head = nn.Sequential(
            nn.Linear(60 * 3 * 3 + self.proprio_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU() if nonlinearity == "relu" else nn.Tanh(),
            nn.Linear(256, self.m),
        )
        self.model = nn.ModuleDict({"neck": neck, "head": head})
        self.device = None

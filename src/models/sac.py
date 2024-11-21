import torch
import torch.nn as nn
import torch.optim as optim
from utils import ReplayBuffer
import ray
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

class SAC:
    def __init__(
        self,
        env_creator,
        num_env_runners=4,
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=0.01,
        gamma=0.95,
        train_batch_size=1024,
        n_step=3,
    ):
        self.env_creator = env_creator
        env_name = "simple_tag_env"
        
        # Register the environment once
        ray.tune.register_env(env_name, lambda config: ParallelPettingZooEnv(self.env_creator()))
        
        # Reference the registered environment ID in SACConfig
        self.config = (
            SACConfig()
            .environment(env=lambda _: ParallelPettingZooEnv(self.env_creator()))
            # .environment(env=env_name)  # Use the registered ID here
            .framework("torch")
            .rollouts(num_env_runners=num_env_runners)
            .training(
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                tau=tau,
                gamma=gamma,
                train_batch_size=train_batch_size,
                n_step=n_step,
            )
            # .rl_module(
            #     # observation_space=self.env_creator().observation_space,
            #     action_space=self.env_creator().action_space,
            #     encoder_config={
            #         "adversary_0": {"type": "fc", "hidden_sizes": [64, 64]},
            #         "agent_0": {"type": "fc", "hidden_sizes": [64, 64]},
            #     }
            # )
        )

    def get_config(self):
        return self.config

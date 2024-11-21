import ray
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import supersuit as ss
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.dqn import DQNConfig
from pettingzoo.mpe import simple_tag_v3
from utils import ReplayBuffer

class DQN:
    def __init__(self,env_creator,num_rollout_workers=4,actor_lr=1e-4,critic_lr=1e-3,tau=0.01,gamma=0.95,train_batch_size=1024,n_step=3):
        
        self.env_creator = env_creator
        env = env_creator()
        env_name = "simple_tag_env"
        
        ray.tune.register_env(env_name, lambda config: ParallelPettingZooEnv(self.env_creator()))
        
        self.config = (
            DQNConfig()
            .environment(env=lambda _: ParallelPettingZooEnv(self.env_creator()))
            # .environment(env=env_name)  # Use the registered ID here
            .framework("torch")
            .rollouts(num_rollout_workers=num_rollout_workers)
            .training(
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                tau=tau,
                gamma=gamma,
                train_batch_size=train_batch_size,
                n_step=n_step,
            )
            .multi_agent(
        policies={agent: (None, env.observation_space(agent), env.action_space(agent), {})
                  for agent in env.possible_agents},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
        )

    def env_creator():
        env = simple_tag_v3.parallel_env(num_good=2, num_adversaries=4, num_obstacles=2, max_cycles=25, continuous_actions=True, render_mode="rgb_array")
        env = ss.pad_observations_v0(env)
        env = ss.pad_action_space_v0(env)
        env = ss.frame_stack_v1(env, 3)
        env = ss.dtype_v0(env, np.float32)  # Ensure observations are float32
        return env

    def get_config(self):
        return self.config

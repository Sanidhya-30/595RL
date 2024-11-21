import numpy as np
import random
from collections import deque
import os
import csv
import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_dim, act_dim, device):
        self.capacity = capacity

        self.obs = np.zeros((capacity, obs_dim))
        self.action = np.zeros((capacity, act_dim))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, done):
        """ add an experience to the memory """
        self.obs[self._index] = obs
        self.action[self._index] = action
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs
        self.done[self._index] = done

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        done = self.done[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        done = torch.from_numpy(done).float().to(self.device)  # just a tensor with length: batch_size

        return obs, action, reward, next_obs, done

    def __len__(self):
        return self._size



class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        states = np.array([exp[0] for exp in experiences], dtype=np.float32)
        actions = np.array([exp[1] for exp in experiences], dtype=np.float32)
        rewards = np.array([exp[2] for exp in experiences], dtype=np.float32)
        next_states = np.array([exp[3] for exp in experiences], dtype=np.float32)
        dones = np.array([exp[4] for exp in experiences], dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Logger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "training_log.csv")
        with open(self.log_file, mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Loss"])

    def log(self, episode, reward, loss):
        with open(self.log_file, mode="a") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, loss])

    def close(self):
        pass


class LinearDecayScheduler:
    def __init__(self, initial_value, final_value, total_steps):
        self.initial_value = initial_value
        self.final_value = final_value
        self.total_steps = total_steps
        self.step = 0

    def get_value(self):
        value = max(
            self.final_value,
            self.initial_value - (self.initial_value - self.final_value) * (self.step / self.total_steps)
        )
        self.step += 1
        return value

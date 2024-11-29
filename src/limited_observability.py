import ray
from pettingzoo.mpe import simple_tag_v3
import numpy as np
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
import gymnasium as gym
from gymnasium import spaces


class LimitedObservabilityWrapper(gym.Wrapper):
    def __init__(self, env, observation_radius=0.5):
        super().__init__(env)
        self.observation_radius = observation_radius

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        return self._modify_observations(observations), info

    def step(self, action):
        observations, rewards, terminations, truncations, info = self.env.step(action)
        return self._modify_observations(observations), rewards, terminations, truncations, info

    def _modify_observations(self, observations):
        modified_obs = {}
        for agent in self.env.possible_agents:
            obs = observations[agent]
            print(obs)
            agent_pos = obs[2:4]  # Indexing the agent's position (the first 2 indices are velocities)
            modified_obs[agent] = self._limit_observation(agent, obs, agent_pos)
        return modified_obs

    def _limit_observation(self, agent, obs, agent_pos):
        limited_obs = obs.copy()
        num_entities = (len(obs) - 4) // 2
        for i in range(num_entities):
            start_idx = 4 + i * 2
            entity_pos = obs[start_idx:start_idx + 2]
            distance = np.linalg.norm(entity_pos - agent_pos)
            if distance > self.observation_radius:
                limited_obs[start_idx:start_idx + 2] = np.zeros(2)

        # Ensure dtype consistency with the original observation space
        return limited_obs.astype(self.env.observation_space(agent).dtype)
    
def env_creator(num_good, num_adversaries, num_obstacles, max_cycles, observation_radius=0.5):
    """Create and preprocess the environment with limited observability."""
    env = simple_tag_v3.parallel_env(
        num_good=num_good,
        num_adversaries=num_adversaries,
        num_obstacles=num_obstacles,
        max_cycles=max_cycles,
        continuous_actions=True,
    )
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.dtype_v0(env, np.float32)  # Ensure observations are float32
    env = LimitedObservabilityWrapper(env, observation_radius)  # Add limited observability
    return env


ray.init(ignore_reinit_error=True,
num_cpus=8)
num_gpus_available = ray.cluster_resources().get("GPU", 0)  
print("Number of GPUS available: ", num_gpus_available) 

# Create and register the environment
env_name = "limited_obs_simple_tag"
tune.register_env(
    env_name, 
    lambda _: ParallelPettingZooEnv(
        env_creator(num_good=2, num_adversaries=4, num_obstacles=2, max_cycles=200)
    )
)

env = env_creator(num_good=2, num_adversaries=4, num_obstacles=2, max_cycles=200)

# Configure the algorithm
config = (
    PPOConfig()
    .environment("limited_obs_simple_tag", env_config={
        "num_good": 2,
        "num_adversaries": 4,
        "num_obstacles": 2,
        "max_cycles": 25,
        "observation_radius": 0.5
        })
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .training(
        train_batch_size=4000,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        sgd_minibatch_size=256,
        num_sgd_iter=10,
    )
    .multi_agent(
        policies=["good", "adversary"],
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: "good" if agent_id.startswith("agent") else "adversary",
    )
)

# Run the training
tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"training_iteration": 200},
    checkpoint_freq=10,
)
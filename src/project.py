import ray
import numpy as np
import scipy
import torch
import sklearn
from ray import tune
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss


def env_creator():
    env = simple_tag_v3.parallel_env(num_good=2, num_adversaries=4, num_obstacles=2, max_cycles=25, continuous_actions=True)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.dtype_v0(env, np.float32)  # Ensure observations are float32
    return env

env = env_creator()

ray.init()

# print("Observation Spaces:")
# for agent in env.possible_agents:
#     print(f"{agent}: {env.observation_space(agent)}")

# print("\nActual Observations:")
# observations = env.reset()
# for agent, obs in observations[0].items():
#     print(f"{agent}: shape={obs.shape}, dtype={obs.dtype}, min={obs.min()}, max={obs.max()}")

env_name = "simple_tag"
tune.register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator()))

config = (
    DDPGConfig()
    .environment(env=env_name)
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .training(
        actor_lr=1e-4,
        critic_lr=1e-3,
        tau=0.01,
        gamma=0.95,
        train_batch_size=1024,
        actor_hiddens=[64, 64],
        critic_hiddens=[64, 64],
        n_step=3,
    )
    .multi_agent(
        policies={agent: (None, env.observation_space(agent), env.action_space(agent), {})
                  for agent in env.possible_agents},
        policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
    )
)

stop = {
    "training_iteration": 500,
    "timesteps_total": 2000000,
    "episode_reward_mean": 200,
}

results = tune.run(
    "DDPG",
    config=config.to_dict(),
    stop=stop,
    checkpoint_freq=10,
    checkpoint_at_end=True,
    local_dir="/local/scratch/a/jshreeku/ece595_reinforcement_learning/src/results", # for some reason, this has to be absoulte paths
    verbose=1,
)

# Get the best trial
best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial final validation reward: {best_trial.last_result['episode_reward_mean']}")
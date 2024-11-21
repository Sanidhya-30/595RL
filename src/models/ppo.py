from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

class PPO:
    def __init__(
        self,
        env_creator,
        num_env_runners=4,
        lr=0.001,
        clip_param=0.2,
        # actor_lr=1e-4,
        # critic_lr=1e-3,
        # tau=0.01,
        # gamma=0.95,
        # train_batch_size=1024,
        # actor_hiddens=[64, 64],
        # critic_hiddens=[64, 64],
        # n_step=3,
    ):
        self.env_creator = env_creator
        self.config = (
            PPOConfig()
            .environment(env=lambda _: ParallelPettingZooEnv(self.env_creator()))
            # .framework("torch")
            .env_runners(num_env_runners=num_env_runners)
            .training(
                # actor_lr=actor_lr,
                # critic_lr=critic_lr,
                lr=lr,
                clip_param=clip_param,
                # gamma=gamma,
                # train_batch_size=train_batch_size,
                # actor_hiddens=actor_hiddens,
                # critic_hiddens=critic_hiddens,
                # n_step=n_step,
            )
        )

    def get_config(self):
        return self.config

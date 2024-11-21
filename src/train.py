import argparse
import os
import ray
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from pettingzoo.mpe import simple_tag_v3
import supersuit as ss
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ddpg import DDPGConfig

def env_creator(num_good, num_adversaries, num_obstacles, max_cycles):
    """Create and preprocess the environment."""
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
    env = ss.dtype_v0(env, "float32")  # Ensure observations are float32
    return env

def get_config(args, env_name):
    """Retrieve the configuration for the selected model."""
    if args.model == "ppo":
        config = (
            PPOConfig()
            .environment(env=env_name)
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_env_runners)
            .training(
                lr=1e-4,
                train_batch_size=args.batch_size,
                gamma=args.gamma,
            )
        )
    elif args.model == "sac":
        config = (
            SACConfig()
            .environment(env=env_name)
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_env_runners)
            .training(
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                tau=args.tau,
                train_batch_size=args.batch_size,
                gamma=args.gamma,
            )
        )
    elif args.model == "dqn":
        config = (
            DQNConfig()
            .environment(env=env_name)
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_env_runners)
            .training(
                lr=args.actor_lr,
                train_batch_size=args.batch_size,
                gamma=args.gamma,
            )
        )
    elif args.model == "ddpg":
        config = (
            DDPGConfig()
            .environment(env=env_name)
            .framework("torch")
            .rollouts(num_rollout_workers=args.num_env_runners)
            .training(
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
                tau=args.tau,
                gamma=args.gamma,
                train_batch_size=args.batch_size,
            )
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    return config

def train(args):
    ray.init(ignore_reinit_error=True)
    
    # Create and register the environment
    env_name = "simple_tag_v3"
    tune.register_env(
        env_name, 
        lambda _: ParallelPettingZooEnv(
            env_creator(args.num_good, args.num_adversaries, args.num_obstacles, args.max_cycles)
        )
    )

    # Get the model configuration
    config = get_config(args, env_name)

    # Set up stop criteria
    stop = {
        "training_iteration": args.training_iterations,
        "timesteps_total": args.timesteps_total,
        "episode_reward_mean": args.reward_threshold,
    }

    # Train the model
    print(f"Training {args.model.upper()} model...")
    # result_dir = os.path.join(args.results_dir, args.model)
    result_dir = os.path.join("/home/lucifer/595RL/src/results", args.model)
    results = tune.run(
        args.model.upper(),
        config=config.to_dict(),
        stop=stop,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=True,
        local_dir=result_dir,
        verbose=1,
    )

    # Save the best checkpoint
    best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    if best_trial:
        best_checkpoint = results.get_best_checkpoint(best_trial, metric="episode_reward_mean", mode="max")
        checkpoint_path = os.path.join(args.checkpoints_dir, f"{args.model}_best_checkpoint")
        os.makedirs(args.checkpoints_dir, exist_ok=True)
        os.rename(best_checkpoint, checkpoint_path)
        print(f"Best checkpoint saved at: {checkpoint_path}")
    else:
        print("No valid trial found. Training may not have completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents on the SimpleTag environment.")
    parser.add_argument("--model", type=str, choices=["ppo", "maddpg", "dqn", "ddpg", "sac"], required=True, help="Model to train.")
    parser.add_argument("--num_good", type=int, default=2, help="Number of good agents.")
    parser.add_argument("--num_adversaries", type=int, default=4, help="Number of adversary agents.")
    parser.add_argument("--num_obstacles", type=int, default=2, help="Number of obstacles.")
    parser.add_argument("--max_cycles", type=int, default=25, help="Max cycles per episode.")
    parser.add_argument("--num_env_runners", type=int, default=4, help="Number of rollout workers.")
    parser.add_argument("--actor_lr", type=float, default=1e-4, help="Learning rate for the actor.")
    parser.add_argument("--critic_lr", type=float, default=1e-3, help="Learning rate for the critic.")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update parameter.")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--training_iterations", type=int, default=500, help="Number of training iterations.")
    parser.add_argument("--timesteps_total", type=int, default=2000, help="Total number of timesteps.")
    parser.add_argument("--reward_threshold", type=float, default=200, help="Reward threshold for stopping training.")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Frequency of checkpoint saving.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save training results.")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory to save checkpoints.")
    args = parser.parse_args()

    train(args)

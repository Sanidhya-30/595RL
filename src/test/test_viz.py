import os
import gymnasium as gym
import numpy as np
from pettingzoo.mpe import simple_tag_v3
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.env import PettingZooEnv
from ray.tune.registry import get_trainable_cls
from moviepy.editor import ImageSequenceClip
import pickle

def load_and_render(model_name, results_dir="/home/lucifer/595RL/src/results", output_dir="/home/lucifer/595RL/src/outputs"):
    """
    Load an RL model from the latest checkpoint and render a video.

    Args:
        model_name (str): The name of the RL algorithm (e.g., 'ddpg', 'ppo').
        results_dir (str): Directory containing the results of training.
        output_dir (str): Directory to save the rendered video.
    """
    model_results_path = os.path.join(results_dir, model_name)
    if not os.path.exists(model_results_path):
        raise FileNotFoundError(f"Results directory for model '{model_name}' not found at {model_results_path}.")

    # Locate the latest experiment folder and checkpoint
    latest_experiment = sorted(os.listdir(model_results_path))[-1]
    experiment_path = os.path.join(model_results_path, latest_experiment)
    checkpoint_path = None

    for root, _, files in os.walk(experiment_path):
        if "checkpoint_000000" in root and "algorithm_state.pkl" in files:
            checkpoint_path = root
            break

    if not checkpoint_path:
        raise FileNotFoundError("No checkpoint directory found in the experiment folder.")

    ray.init(ignore_reinit_error=True)

    # Load configuration from params.pkl
    config_path = os.path.join(experiment_path, "params.pkl")
    config_path = "/home/lucifer/595RL/src/results/ddpg/DDPG_2024-11-18_19-55-36/DDPG_simple_tag_fcdfd_00000_0_2024-11-18_19-55-36/params.pkl"
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    # Load the RL algorithm and restore the checkpoint
    algo_cls = get_trainable_cls(model_name.upper())
    algorithm = algo_cls(config=config)
    algorithm.restore(checkpoint_path)

    # Prepare the environment
    env = PettingZooEnv(simple_tag_v3.env(max_cycles=25))
    env.reset()

    # Render and save the video
    frames = []
    done = False
    while not done:
        actions = {agent: algorithm.compute_single_action(env.observe(agent)) for agent in env.agents}
        _, _, done, _, _ = env.step(actions)
        frames.append(env.render(mode="rgb_array"))

    output_video_path = os.path.join(output_dir, model_name, os.path.basename(checkpoint_path))
    os.makedirs(output_video_path, exist_ok=True)
    video_file = os.path.join(output_video_path, "render.mp4")

    clip = ImageSequenceClip(frames, fps=30)
    clip.write_videofile(video_file, codec="libx264")

    print(f"Video saved at {video_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render and save RL model video.")
    parser.add_argument("model_name", type=str, help="The name of the RL algorithm (e.g., 'ddpg', 'ppo').")
    args = parser.parse_args()

    load_and_render(args.model_name)

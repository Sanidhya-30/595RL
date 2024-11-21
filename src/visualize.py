import argparse
import cv2
import os
import json
import numpy as np
import time
from pathlib import Path
from importlib import import_module
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.algorithms.ddpg import DDPG, DDPGConfig
from pettingzoo.mpe import simple_tag_v3
import sys




frame_dir = "/home/lucifer/595RL/src/outputs/frame"
video_dir = "/home/lucifer/595RL/src/outputs/video"
if not os.path.exists(frame_dir):
    os.makedirs(frame_dir)
    os.makedirs(video_dir)



class CustomParallelPettingZooEnv(ParallelPettingZooEnv):
    def __init__(self, env):
        super().__init__(env)
        self.env = env  # Store the original environment

    def render(self, mode='human'):
        return self.env.render()  # Pass the mode parameter


def load_checkpoint(checkpoint_path, config):
    algo = DDPG(config=config)
    algo.restore(checkpoint_path)
    return algo


def save_rendered_frame(frame, frame_count):
    # Convert frame from RGB to BGR for OpenCV compatibility
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Save the frame as an image file
    frame_file = f'{frame_dir}/frame_{frame_count}.png'
    cv2.imwrite(frame_file, frame)


def save_frame(obs, frame_count):
    for agent, observation in obs.items():
        # Ensure the observation is in the correct format (HxWxC)
        if observation.ndim == 3 and observation.shape[0] == 3:  # If CxHxW format
            frame = observation.transpose(1, 2, 0)
        else:
            frame = observation

        # Normalize and convert to uint8 if necessary
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Ensure the frame is in RGB format (OpenCV uses BGR)
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame_file = f'{frame_dir}/frame_{frame_count}_{agent}.png'
        cv2.imwrite(frame_file, frame)

    return frame_count + 1


def render_environment(env, algo, num_episodes=5, video_filename="output_video.mp4", fps=10):
    frame_count = 0
    
    # Initialize video writer
    video_writer = None
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = {agent: False for agent in obs.keys()}
        episode_reward = 0
        step = 0
        
        while not all(done.values()):
            actions = {agent: algo.compute_single_action(obs[agent], policy_id=agent) for agent in obs.keys()}
            obs, rewards, terminated, truncated, _ = env.step(actions)
            done = {agent: terminated[agent] or truncated[agent] for agent in obs.keys()}
            episode_reward += sum(rewards.values())
            
            # Capture and save the rendered frame
            rendered_frame = env.render(mode='rgb_array')  # Get pixel array from render()
            
            if rendered_frame is not None:
                # Save individual frames as images
                save_rendered_frame(rendered_frame, frame_count)
                
                # Initialize VideoWriter if it's not already initialized
                if video_writer is None:
                    height, width, layers = rendered_frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
                    video_writer = cv2.VideoWriter(f"{video_dir}/{video_filename}", fourcc, fps, (width, height))
                
                # Write the frame to the video file
                video_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
                
                frame_count += 1
            
            time.sleep(0.1)
            step += 1
        
        print(f"Episode {episode + 1} finished after {step} steps. Total reward: {episode_reward}")
    
    env.close()
    
    # Release the VideoWriter to finalize the video file
    if video_writer is not None:
        video_writer.release()
    
    print(f"Frames saved in {frame_dir}")
    print(f"Video saved as {video_filename}")


def get_latest_checkpoint(model):
    """Find the latest checkpoint for the specified model in the results directory."""
    model_dir = Path(f"./results/{model}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")
    
    # Find the latest experiment folder based on its name
    experiment_dirs = list(model_dir.glob(f"{model.upper()}_*"))
    if not experiment_dirs:
        raise FileNotFoundError(f"No experiments found for model {model} in {model_dir}.")
    
    latest_experiment = max(experiment_dirs, key=lambda x: x.name)
    
    # Locate the checkpoint folder inside the latest experiment
    checkpoint_dirs = list(latest_experiment.glob(f"{model.upper()}_*/checkpoint_*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in the latest experiment {latest_experiment}.")
    
    # Return the latest checkpoint directory
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: x.name)
    return latest_checkpoint


def get_latest_config(config_dir):
    """Find the latest configuration file based on the file name."""
    config_files = list(Path(config_dir).glob("*.json"))
    if not config_files:
        raise FileNotFoundError(f"No configuration files found in {config_dir}")
    latest_config = max(config_files, key=lambda x: x.name)
    return latest_config


def get_env_config_from_checkpoint(checkpoint_path):
    """Extract environment configuration from `rllib_checkpoint.json`."""
    import json
    from pathlib import Path

    rllib_checkpoint_file = Path(checkpoint_path) / "rllib_checkpoint.json"
    if not rllib_checkpoint_file.exists():
        raise FileNotFoundError(f"{rllib_checkpoint_file} not found")

    with open(rllib_checkpoint_file, "r") as f:
        checkpoint_data = json.load(f)

    # Extract relevant fields for environment configuration
    policy_ids = checkpoint_data.get("policy_ids", [])
    if not policy_ids:
        raise ValueError("Policy IDs not found in checkpoint data")

    # Create a default environment configuration based on policy IDs
    env_config = {
        "num_good": sum("agent" in pid for pid in policy_ids),
        "num_adversaries": sum("adversary" in pid for pid in policy_ids),
        "num_obstacles": 2,  # Default or inferred value
        "max_cycles": 25,  # Default or inferred value
        "continuous_actions": True,  # Default or inferred value
        "render_mode": "rgb_array",  # Default or inferred value
    }

    return env_config



def env_creator(env_config):
    """Create an environment using the extracted config."""
    import supersuit as ss
    from pettingzoo.mpe import simple_tag_v3

    env = simple_tag_v3.parallel_env(**env_config)
    env = ss.pad_observations_v0(env)
    env = ss.pad_action_space_v0(env)
    env = ss.frame_stack_v1(env, 3)
    env = ss.dtype_v0(env, np.float32)  # Ensure observations are float32
    return env


def main():
    parser = argparse.ArgumentParser(description="Visualize the environment and save outputs.")
    parser.add_argument("--model", required=True, help="Name of the model to use (e.g., 'ddpg')")
    parser.add_argument("--config", default=None, help="Path to the configuration file for the model.")
    parser.add_argument("--video_dir", default="./outputs/video", help="Directory to save video files.")
    parser.add_argument("--image_dir", default="./outputs/frames", help="Directory to save image frames.")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to render.")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for the output video.")
    args = parser.parse_args()

    sys.path.append(str(Path(__file__).parent))
    model_dir = f"./src/results/{args.model.lower()}"
    frame_dir = args.image_dir
    video_dir = args.video_dir

    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    try:
        latest_checkpoint_path = str(get_latest_checkpoint(args.model))
        # latest_checkpoint_path = get_latest_checkpoint(model_dir)
        print(f"Latest checkpoint found: {latest_checkpoint_path}")
    except FileNotFoundError as e:
        print(e)
        return
    
    if args.config is None:
        # config_dir = f"./src/config"
        try:
            args.config = get_latest_config(latest_checkpoint_path)
            print(f"Using latest configuration file: {args.config}")
        except FileNotFoundError as e:
            print(e)
            return


    env_config = get_env_config_from_checkpoint(latest_checkpoint_path)
    print(f"Environment configuration loaded: {env_config}")

    # loaded_algo = load_checkpoint(latest_checkpoint_path, config)
    env_name = "simple_tag_v3"
    env = env_creator(env_config)
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



    algo = DDPG(config=config)
    algo.restore(latest_checkpoint_path)
    render_env = CustomParallelPettingZooEnv(env_creator(env_config))
    render_environment(frame_dir, video_dir, render_env, algo, args.num_episodes, f"{args.model}_output.mp4", args.fps)


if __name__ == "__main__":
    main()

# 595RL: UAV Swarm Tracking With Multi-Agent RL

This repository contains code and resources for a project focused on multi agent object tracking and surveillance using deep reinforcement learning. The project is implemented in Python 3.11 and leverages key libraries such as Ray and PettingZoo.

The following commands are for linux based system, for windows please create a conda environment for the project.

---

## Installation

### Requirements
1. Install **Python 3.11.7**. You can download it from [python.org](https://www.python.org).
2. Clone this repository:
   ```bash
   git clone https://github.com/JayanthShreekumar/595RL
   cd 595RL
   git checkout Tag_San
   ```

### Dependencies and Virtual Environment
Install the required Python packages using `requirements.txt`:
```bash
python -m venv .rl_venv
source .rl_venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Key dependencies:
- dm-tree==0.1.8
- gymnasium==0.28.1
- matplotlib==3.9.2
- numpy==1.24.3
- pandas==2.2.3
- pettingzoo==1.24.3
- pyarrow==18.0.0
- ray==2.8.0
- ray[tune]
- scikit-image==0.24.0
- SuperSuit==3.9.3
- tensorboard==2.18.0
- tensorboard-data-server==0.7.2
- tensorboardX==2.6.2


### Environment Changes

Change the contents of your simple tag environment file located at your python3 site packages

```bash
cd ~/${user}/.local/lib/python3.xx
cd site-packages/pettingzoo/mpe/simple_tag
gedit simple_tag.py
```

with the contents of simple_tag_changes.py



---

## Repository Structure

- **`src/`**: Contains the main implementation of the project, including training scripts.
- **`train.py`**: Script to train the environment with various configuration/models
- **`plotting_results.ipynb`**: A Jupyter notebook for visualizing the performance metrics and results.
- **`visualize_results.ipynb`**: A Jupyter notebook for exploring the model's behavior and intermediate results.
- **`requirements.txt`**: Specifies all Python dependencies for the project.
- **`sample_sbatch.sh`**: Example of how code was queued to run on high-performance cluster



---



## Usage

### 1. Training the Model
To train the model, use the `train.py` script. Navigate to the root directory (or where the `src/` directory is located) and execute:

```bash
cd ~/595RL
python src/train.py --model <model-name> [other arguments]
```
Note that framework assumes access to at least 1 gpu and 1 cpus to complete training. If this is not the case the source code must be adjusted to use fewer resources. `sample_sbatch.sh` shows how this script was queued to run with designated resources.

#### Required Argument:
- `--model`: The RL algorithm to train. Options: `ppo`, `ddpg`, `sac`, `single_sac`.

#### Optional Arguments:
- `--num_good`: Number of good agents. Default: `2`.
- `--num_adversaries`: Number of adversary agents. Default: `4`.
- `--num_obstacles`: Number of obstacles. Default: `0`.
- `--max_cycles`: Max cycles per episode. Default: `100`.
- `--obs_radius`: Radius of observability for all agents bw [0.0, 1.0]. Default: `0.0`.
- `--num_env_runners`: Number of rollout workers. Default: `7`.
- `--actor_lr`: Learning rate for the actor. Default: `0.001`.
- `--critic_lr`: Learning rate for the critic. Default: `0.005`.
- `--tau`: Soft update parameter. Default: `0.01`.
- `--gamma`: Discount factor. Default: `0.95`.
- `--batch_size`: Training batch size. Default: `1024`.
- `--training_iterations`: Number of training iterations. Default: `500`.
- `--timesteps_total`: Total number of timesteps. Default: `2000`.
- `--reward_threshold`: Reward threshold for stopping training. Default: `200`.
- `--checkpoint_freq`: Frequency of checkpoint saving. Default: `10`.
- `--results_dir`: Directory to save training results. Default: `results`.
- `--checkpoints_dir`: Directory to save checkpoints. Default: `checkpoints`.

#### Example Command:
```bash
python train.py --model ppo --num_good 3 --num_adversaries 5 --batch_size 2048
```

![Train](https://github.com/Sanidhya-30/595RL/blob/main/media/traianing_dist.jpg?raw=true)


---


### 2. Visualizing Results
Open the `visualize_results.ipynb` notebook in JupyterLab or Jupyter Notebook:
change the checkpoint_path to your respective results path

```bash
jupyter notebook visualize_results.ipynb
```

This notebook helps analyze the intermediate results of the model during training.

![Viz](https://github.com/Sanidhya-30/595RL/blob/main/media/RL.gif)
---

### 3. Plotting Performance Metrics
To generate performance plots, use the `plotting_results.ipynb` notebook. Launch it using:
change the checkpoint_path to your respective results path

```bash
jupyter notebook plotting_results.ipynb
```

![Eval](https://github.com/Sanidhya-30/595RL/blob/main/media/trainingPPO.jpg?raw=true)


<!--![PPO](https://github.com/Sanidhya-30/595RL/blob/main/media/train_ppo.jpg?raw=true)-->

---

## Find out the report at [link](https://github.com/Sanidhya-30/595RL/blob/main/media/ECE595___Reinforcement_Learning_Project.pdf)

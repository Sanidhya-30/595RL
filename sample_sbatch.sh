#!/bin/bash 
#SBATCH -J train_simple_tag
#SBATCH -N1 --mem=50GB
#SBATCH --time=0-04:00:00  # 2 hours
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH -o %u-%x-job%j.out
#SBATCH --export=ALL


# Print out Job Details
echo "Job ID: "$SLURM_JOB_ID
nvidia-smi -L
echo "CUDA VISIBLE DEVICES: $CUDA_VISIBLE_DEVICES"
echo "------------"

# Run Python Code
echo "Executing Python Code:"
echo "this code has a distance based reward function"
echo "this code has a continuous reward"


python3 src/train.py --model "single_sac" --num_env_runners 7 --training_iterations 3000 --max_cycles 100 --checkpoint_freq 100

# Print Environment Variables to File
env > environment.txt

echo "Job completed."
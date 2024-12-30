#! /bin/bash

#SBATCH --job-name=CoSOD
#SBATCH --output=output_a32.%A.txt # Standard output and error.
#SBATCH --error=error_a32.%A.err # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=64G # Total RAM to be used
#SBATCH --cpus-per-task=1 # Number of CPU cores
#SBATCH --gres=gpu:1 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos

#python train.py
python test4.py

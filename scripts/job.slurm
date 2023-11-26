#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=80G
#SBATCH --job-name="GigaSemSegV100"
#SBATCH --output=./scripts/jupyterA100.out
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80g

module load Anaconda3/2022.05
jupyter notebook --no-browser
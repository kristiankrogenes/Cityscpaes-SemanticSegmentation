#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --ntasks-per-node=1
#SBATCH --job-name="Gigatron"
#SBATCH --output=joblog.out
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=gpu32g

module purge
module load Anaconda3/2022.05
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1

python main.py
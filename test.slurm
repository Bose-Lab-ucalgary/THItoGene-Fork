#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=thitogene-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100

####### Run your script #########################
source ~/software/init-conda
conda activate thitogene-env


python inference.py 
# python train-test-HEST.py --gpus 2 --batch_size 2 --precision 16 --num_workers 8
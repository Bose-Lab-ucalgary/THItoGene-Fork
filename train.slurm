#!/bin/bash

####### Reserve computing resources #############
#SBATCH --job-name=thitogene
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=148G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100

####### Run your script #########################
source ~/software/init-conda
conda activate thitogene-env


python train.py
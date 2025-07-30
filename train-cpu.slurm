#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu2023

####### Run your script #########################
source ~/software/init-conda
conda activate thitogene-env


python train.py
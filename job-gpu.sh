#!/bin/bash

#SBATCH -J inference             # Job name
#SBATCH -p small                # Partition name
#SBATCH --nodes=1                # Minimum number of nodes
#SBATCH --gres=gpu:1             # Number of GPUs per node
#SBATCH --ntasks 1               # no of tasks
#SBATCH --time=24:00:00           # hh:mm:ss

# Bind the required directories and run the training script
dirname="Llama-3.1-70B-Instruct/"
mkdir /tmpdir/$USER/$dirname
cd /tmpdir/$USER/$dirname


apptainer exec --bind /tmpdir,/work --nv /work/conteneurs/sessions-interactives/pytorch-24.02-py3-calmip-si.sif python /tmpdir/m24047krsh/llama_project/llama_inference/code/generate_answer.py

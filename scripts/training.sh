#!/bin/bash
#SBATCH --job-name=train
#SBATCH --nodes=1  
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=fail  
#SBATCH --mail-user=tt1131@princeton.edu
#SBATCH --output=/usr/people/tt1131/projects/MMMmB/qmc_deep_gen/scripts/out/slurm-%A_%a.out

module purge
module load anacondapy/2023.07
conda activate /usr/people/tt1131/.conda/envs/samv2_env

python /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/bartul_mouse.py \
    --save_location /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/results/mouse_150k_2D \
    --dataloc /jukebox/falkner/Dexter/vocal_beh/data/full_dataset/preprocessed_500k_110_centered_512/ \
    --nEpochs 300 \
    --max_train_samples 150000 \
    --train_batch_size 512 \
    --test_batch_size 1 \
    --print_gpu_mem True \
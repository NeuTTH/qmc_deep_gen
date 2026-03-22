#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --nodes=1  
#SBATCH --ntasks=1    
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=fail  
#SBATCH --mail-user=tt1131@princeton.edu
#SBATCH --output=/usr/people/tt1131/projects/MMMmB/qmc_deep_gen/scripts/out/slurm-%A_%a.out

module purge
module load anacondapy/2023.07
conda activate /usr/people/tt1131/.conda/envs/samv2_env

python /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/analyze_mouse_latents.py \
    --model_path /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/results/mouse_test_small_100k_binary_fixed_no_cond/qmc_train_mouse_experiment.tar \
    --dataloc /jukebox/falkner/Dexter/vocal_beh/data/full_dataset/preprocessed_500k_110_centered_512/ \
    --save_dir /usr/people/tt1131/projects/MMMmB/qmc_deep_gen/results/mouse_test_small_100k_binary_fixed_no_cond/latents \
    --lattice_m 25 \
    --bandwidth 0.1 \
    --batch_size 64 \
#!/bin/bash
#SBATCH -J mae3d_vit_base_adni_mr80_ps8_bs8
#SBATCH -N 1
#SBATCH -p mcml-hgx-h100-92x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=256gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH -o %x.%j.%N.out

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate mae # activate your environment

cd ~/SelfMedMAE/

export WANDB_API_KEY=9b379393a7a65969e05ab4e01683be3b8770aabf

srun python main.py \
     configs/mae3d_adni_1gpu.yaml \
     --patch_size=8 \
     --mask_ratio=0.80 \
     --batch_size=8 \
     --run_name='mae3d_vit_base_adni_mr80_ps8_bs8'
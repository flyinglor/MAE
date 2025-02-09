#!/bin/bash
#SBATCH -J mae3d_vit_large_ukb_mr80_ps16_bs32_ep1000_lr1e3
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --nodelist=mcml-hgx-a100-015
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=512gb
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
     configs/mae3d_ukb_1gpu.yaml \
     --patch_size=16 \
     --mask_ratio=0.80 \
     --batch_size=32 \
     --run_name='mae3d_vit_large_ukb_mr80_ps16_bs32_ep1000_lr1e3'
#!/bin/bash
#SBATCH -J encoder_3fc_ukb_hos_bs8_ep1000_lr0.001
#SBATCH -N 1
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --mem=128gb
#SBATCH --ntasks=1
#SBATCH --mail-user=hui.zheng@tum.de
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH -o %x.%j.%N.out

source ~/.bashrc  # activate miniconda
source ~/miniconda3/bin/activate mae # activate your environment

cd ~/SelfMedMAE/

export WANDB_API_KEY=9b379393a7a65969e05ab4e01683be3b8770aabf


srun python main.py \
        configs/mae3d_finetune_1gpu.yaml \
        --proj_name=MAE3DFINETUNE \
        --batch_size=8 \
        --epochs=1000 \
        --run_name=encoder_3fc_ukb_hos_bs8_ep1000_lr0.001 \
        --pretrain=/dss/dsshome1/0C/ge79qex2/SelfMedMAE/pretrained_weights/ukb_mr80_checkpoint_0999.pth.tar
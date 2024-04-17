#!/bin/bash
#SBATCH -o job_output/job.%j.out
#SBATCH -p titan
#SBATCH --qos=titan
#SBATCH -J Anonymize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --exclude=rtxgpu002
#SBATCH --gres=gpu:4

source activate anon2

nvidia-smi

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
--nproc_per_node=4 --master_port=10994 \
stage1_train_anon_inpaint_lora_version.py \
--launcher=pytorch \
--config configs/training/anon/stage1_image_finetune_inpaint.yaml \
--use_8bit_adam
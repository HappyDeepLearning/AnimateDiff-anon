#!/bin/bash
#SBATCH -o job_output/job.%j.out
#SBATCH -p a100
#SBATCH --qos=a100
#SBATCH -J bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4

source activate anon2

nvidia-smi


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
--nproc_per_node=4 --master_port=10994 \
stage1_train_anon_inpaint_lora_version.py \
--launcher=pytorch \
--config configs/training/anon/stage1_image_finetune_inpaint.yaml \
--use_8bit_adam

# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
# --nproc_per_node=4 --master_port=10994 \
# train_anon_inpaint_controlnet.py \
# --launcher=pytorch \
# --config configs/training/v1/image_finetune_inpaint_controlnet_a100.yaml
# stage 1
# NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
# --nproc_per_node=4 --master_port=10994 stage1_train_anon_inpaint_lora_version.py \
# --config configs/training/anon/stage1_image_finetune_inpaint.yaml 

# # stage 2
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
--nproc_per_node=4 --master_port=10994 stage2_train_anon_inpaint.py \
--config configs/training/anon/stage2_training.yaml


# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
# --nproc_per_node=4 --master_port=10994 train_anon_inpaint_controlnet.py \
# --config configs/training/v1/training_controlnet.yaml

# NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun  \
# --nproc_per_node=4 --master_port=10994 train_anon_inpaint_controlnet.py \
# --config configs/training/v1/training_controlnet.yaml
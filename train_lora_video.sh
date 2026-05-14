#!/bin/bash
SHARED_CACHE_DIR=/cpfs01/cpfs01/cache/modelscope
mkdir -p $SHARED_CACHE_DIR/datasets/map_cache

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=401408 \
MIN_PIXELS=3136 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=768 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=/cpfs01/jensen/code/code_latest/ms-swift \
ROOT_IMAGE_DIR=/cpfs01/cpfs01/datas/vlm_dataset \
MODELSCOPE_CACHE=$SHARED_CACHE_DIR \
HF_DATASETS_CACHE=$SHARED_CACHE_DIR/datasets \
.venv/bin/python -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port=29500 \
    --master_addr=127.0.0.1 \
    swift/cli/sft.py \
    --model /cpfs01/cpfs01/models/Qwen3.6-27B \
    --dataset /cpfs01/jensen/code/code_latest/ms-swift/output/subtask_dataset_30with_70without_unique.json \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --max_length 20480 \
    --dataset_num_proc 32 \
    --deepspeed zero2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --dataset_shuffle true \
    --dataloader_num_workers 8 \
    --load_from_cache_file true \
    --packing true \
    --attn_impl flash_attn \
    --output_dir output/qwen3_6_lora_video

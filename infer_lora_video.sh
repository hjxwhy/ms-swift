#!/bin/bash
set -e

SHARED_CACHE_DIR=/cpfs01/cpfs01/cache/modelscope
mkdir -p $SHARED_CACHE_DIR/datasets/map_cache

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/cpfs01/jensen/code/code_latest/ms-swift/output/qwen3_6_lora_video/v15-20260512-143145/checkpoint-300}
DATASET=${DATASET:-/cpfs01/jensen/code/code_latest/ms-swift/output/subtask_dataset_30with_70without_unique.json}
SAMPLE_SIZE=${SAMPLE_SIZE:-1}
RESULT_PATH=${RESULT_PATH:-/cpfs01/jensen/code/code_latest/ms-swift/output/qwen3_6_lora_video/infer_checkpoint_300_sample${SAMPLE_SIZE}.jsonl}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=401408 \
MIN_PIXELS=3136 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=768 \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
PYTHONPATH=/cpfs01/jensen/code/code_latest/ms-swift \
ROOT_IMAGE_DIR=/cpfs01/cpfs01/datas/vlm_dataset \
MODELSCOPE_CACHE=$SHARED_CACHE_DIR \
HF_DATASETS_CACHE=$SHARED_CACHE_DIR/datasets \
.venv/bin/python swift/cli/infer.py \
    --adapters $CHECKPOINT_DIR \
    --dataset $DATASET \
    --split_dataset_ratio 1.0 \
    --dataset_num_proc 32 \
    --load_from_cache_file true \
    --dataset_shuffle true \
    --infer_backend transformers \
    --val_dataset_sample $SAMPLE_SIZE \
    --max_length 262144 \
    --max_new_tokens 16384 \
    --temperature 0.5 \
    --stream false \
    --result_path $RESULT_PATH \
    --truncation_strategy right

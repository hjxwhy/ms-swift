SHARED_CACHE_DIR=/cpfs01/cpfs01/cache/modelscope
DATASET_DIR=/cpfs01/cpfs01/datas/vlm_dataset/pretrain_vlm_action_mask_tokens_hf
# Registered as `pretrain-vlm-action` (swift/dataset/dataset/vla.py) with subsets:
#   action gpm idm it2vm sp vpm
# `all` expands (via _select_subsets) to every registered subset and concatenates them.
# The local multi-config dir must be addressed by path:subset (not the registered name,
# which the loader would force onto the single-config path loader); the dir matches the
# registered dataset_path, so the subset list still comes from vla.py.
SUBSET=${SUBSET:-all}
OUTPUT_DIR=$SHARED_CACHE_DIR/cached_datasets/Qwen3-VL-4B-Instruct-pretrain_vlm_action-${SUBSET}-1024
mkdir -p $SHARED_CACHE_DIR/datasets/map_cache

# --- CPU thread caps: must be set BEFORE python/torch starts ---
export OMP_NUM_THREADS=4          # main lever (controls ATen / MKL via OpenMP)
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4   # harmless on Linux, useful on macOS

# Optional but recommended on big NUMA boxes:
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
# export OMP_PROC_BIND=close
# export OMP_PLACES=cores

# export FORCE_QWENVL_VIDEO_READER=decord

# Export the qwen3vl general parquet dataset into a cached HF dataset.
# Parquet files already use the standard {messages, images} schema; the
# `id` column is dropped automatically by remove_unused_columns.
#
# Threading env vars are critical: without them, each of the
# `--dataset_num_proc` workers spins up O(num_cores) torch CPU
# threads inside the HF image processor, causing massive contention
# (~300x slowdown on Qwen2/3-VL preprocessing). One thread per worker
# lets the multiprocess pool scale linearly.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1048576 \
MIN_PIXELS=3136 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=128 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=/cpfs01/jensen/code/code_latest/ms-swift \
ROOT_IMAGE_DIR=/cpfs01/cpfs01/datas/vlm_dataset \
MODELSCOPE_CACHE=$SHARED_CACHE_DIR \
HF_DATASETS_CACHE=$SHARED_CACHE_DIR/datasets \
.venv/bin/python swift/cli/export.py \
    --model /cpfs01/cpfs01/models/Qwen3-VL-4B-Base/ \
    --dataset ${DATASET_DIR}:${SUBSET} \
    --split_dataset_ratio 0 \
    --dataset_num_proc 128 \
    --max_length 8192 \
    --to_cached_dataset True \
    --output_dir $OUTPUT_DIR \
    --exist_ok True \
    --robot_state_dim 120 \
    --new_special_tokens examples/continue_pretrain_vla/vlm_mask_special_tokens_remove_state.txt

#!/bin/bash
# Parallel evaluation script for comparing models
# Usage: ./scripts/eval_parallel.sh

set -e

# Configuration
TEST_DIR="./data/voicebank-demand/test"
CLEAN_DIR="./data/voicebank-demand/test/clean"
NOISY_DIR="./data/voicebank-demand/test/noisy"

# Checkpoints
NOISE_COND_CKPT="./lightning_logs/version_0/checkpoints/epoch=138-step=50000.ckpt"
BASELINE_CKPT="./checkpoints/sgmse_vb_demand.ckpt"

# Output directories
OUTPUT_NOISE_COND="./enhanced_noise_cond"
OUTPUT_BASELINE="./enhanced_baseline"

# GPU configuration
GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# Number of diffusion steps
N_STEPS=30

echo "=========================================="
echo "Parallel Evaluation Script"
echo "=========================================="
echo "GPUs: ${GPUS[*]}"
echo "Noise-cond checkpoint: $NOISE_COND_CKPT"
echo "Baseline checkpoint: $BASELINE_CKPT"
echo ""

# Download baseline checkpoint if not exists
if [ ! -f "$BASELINE_CKPT" ]; then
    echo "Downloading baseline SGMSE+ checkpoint..."
    mkdir -p ./checkpoints
    gdown 1_H3EXvhcYBhOZ9QNUcD5VZHc6ktrRbwQ -O "$BASELINE_CKPT"
fi

# Get list of test files and split
NOISY_FILES=($(ls "$NOISY_DIR"/*.wav 2>/dev/null))
TOTAL_FILES=${#NOISY_FILES[@]}
FILES_PER_GPU=$((TOTAL_FILES / NUM_GPUS))

echo "Total test files: $TOTAL_FILES"
echo "Files per GPU: ~$FILES_PER_GPU"
echo ""

# Create temp directories for split files
TEMP_DIR="./temp_eval_splits"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    mkdir -p "$TEMP_DIR/split_$i/noisy"
    mkdir -p "$TEMP_DIR/split_$i/clean"
done

# Split files into groups
echo "Splitting files across GPUs..."
for i in "${!NOISY_FILES[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    FILENAME=$(basename "${NOISY_FILES[$i]}")
    ln -s "$(realpath "$NOISY_DIR/$FILENAME")" "$TEMP_DIR/split_$GPU_IDX/noisy/$FILENAME"
    ln -s "$(realpath "$CLEAN_DIR/$FILENAME")" "$TEMP_DIR/split_$GPU_IDX/clean/$FILENAME"
done

# Create output directories
mkdir -p "$OUTPUT_NOISE_COND"
mkdir -p "$OUTPUT_BASELINE"

echo ""
echo "=========================================="
echo "Running Noise-Conditioned Model"
echo "=========================================="

# Run noise-cond model in parallel
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU=${GPUS[$i]}
    echo "Starting GPU $GPU for split $i..."
    CUDA_VISIBLE_DEVICES=$GPU python enhancement_noise_cond.py \
        --test_dir "$TEMP_DIR/split_$i" \
        --enhanced_dir "$OUTPUT_NOISE_COND" \
        --ckpt "$NOISE_COND_CKPT" \
        --oracle_noise \
        --clean_dir "$TEMP_DIR/split_$i/clean" \
        --N $N_STEPS \
        --device cuda &
done
wait
echo "Noise-cond model done!"

echo ""
echo "=========================================="
echo "Running Baseline SGMSE+ Model"
echo "=========================================="

# Run baseline model in parallel
for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU=${GPUS[$i]}
    echo "Starting GPU $GPU for split $i..."
    CUDA_VISIBLE_DEVICES=$GPU python enhancement.py \
        --test_dir "$TEMP_DIR/split_$i/noisy" \
        --enhanced_dir "$OUTPUT_BASELINE" \
        --ckpt "$BASELINE_CKPT" \
        --N $N_STEPS \
        --device cuda &
done
wait
echo "Baseline model done!"

# Cleanup temp directory
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "Computing Metrics"
echo "=========================================="

echo ""
echo "--- Noise-Conditioned Model ---"
python calc_metrics.py --test_dir "$TEST_DIR" --enhanced_dir "$OUTPUT_NOISE_COND"

echo ""
echo "--- Baseline SGMSE+ ---"
python calc_metrics.py --test_dir "$TEST_DIR" --enhanced_dir "$OUTPUT_BASELINE"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="

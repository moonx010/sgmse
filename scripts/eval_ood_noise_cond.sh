#!/bin/bash
# OOD evaluation for noise-conditioned model (vb_esc50)
# Runs 3 SNR levels in parallel on different GPUs

set -e

CKPT="./lightning_logs/version_0/checkpoints/epoch=138-step=50000.ckpt"
N_STEPS=30

echo "Running noise-cond enhancement on vb_esc50..."

# Run all 3 SNR levels in parallel on different GPUs
CUDA_VISIBLE_DEVICES=0 python enhancement_noise_cond.py --test_dir ./data/test_mixtures/vb_esc50/snr_0dB --enhanced_dir ./enhanced_ood_nc_0dB --ckpt "$CKPT" --oracle_noise --clean_dir ./data/test_mixtures/vb_esc50/snr_0dB/clean --N $N_STEPS --device cuda &

CUDA_VISIBLE_DEVICES=1 python enhancement_noise_cond.py --test_dir ./data/test_mixtures/vb_esc50/snr_5dB --enhanced_dir ./enhanced_ood_nc_5dB --ckpt "$CKPT" --oracle_noise --clean_dir ./data/test_mixtures/vb_esc50/snr_5dB/clean --N $N_STEPS --device cuda &

CUDA_VISIBLE_DEVICES=2 python enhancement_noise_cond.py --test_dir ./data/test_mixtures/vb_esc50/snr_10dB --enhanced_dir ./enhanced_ood_nc_10dB --ckpt "$CKPT" --oracle_noise --clean_dir ./data/test_mixtures/vb_esc50/snr_10dB/clean --N $N_STEPS --device cuda &

wait
echo "Enhancement done!"

echo ""
echo "Computing metrics..."

echo "=== SNR 0dB ==="
python calc_metrics.py --clean_dir ./data/test_mixtures/vb_esc50/snr_0dB/clean --noisy_dir ./data/test_mixtures/vb_esc50/snr_0dB/noisy --enhanced_dir ./enhanced_ood_nc_0dB

echo ""
echo "=== SNR 5dB ==="
python calc_metrics.py --clean_dir ./data/test_mixtures/vb_esc50/snr_5dB/clean --noisy_dir ./data/test_mixtures/vb_esc50/snr_5dB/noisy --enhanced_dir ./enhanced_ood_nc_5dB

echo ""
echo "=== SNR 10dB ==="
python calc_metrics.py --clean_dir ./data/test_mixtures/vb_esc50/snr_10dB/clean --noisy_dir ./data/test_mixtures/vb_esc50/snr_10dB/noisy --enhanced_dir ./enhanced_ood_nc_10dB

echo ""
echo "Done!"

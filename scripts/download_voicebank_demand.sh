#!/bin/bash
# Download and prepare VoiceBank-DEMAND dataset for SGMSE+
# Source: https://datashare.ed.ac.uk/handle/10283/2791

set -e

# Default output directory
OUTPUT_DIR="${1:-./data/voicebank-demand}"

echo "=========================================="
echo "VoiceBank-DEMAND Dataset Download Script"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create directory structure
mkdir -p "$OUTPUT_DIR"/{train,valid,test}/{clean,noisy}
mkdir -p "$OUTPUT_DIR/downloads"

cd "$OUTPUT_DIR/downloads"

# Download files
echo "[1/4] Downloading clean trainset..."
wget -c https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip

echo "[2/4] Downloading noisy trainset..."
wget -c https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip

echo "[3/4] Downloading clean testset..."
wget -c https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip

echo "[4/4] Downloading noisy testset..."
wget -c https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip

# Extract and organize
echo ""
echo "Extracting files..."

echo "Extracting clean trainset..."
unzip -q -o clean_trainset_28spk_wav.zip
mv clean_trainset_28spk_wav/* "$OUTPUT_DIR/train/clean/"
rmdir clean_trainset_28spk_wav

echo "Extracting noisy trainset..."
unzip -q -o noisy_trainset_28spk_wav.zip
mv noisy_trainset_28spk_wav/* "$OUTPUT_DIR/train/noisy/"
rmdir noisy_trainset_28spk_wav

echo "Extracting clean testset..."
unzip -q -o clean_testset_wav.zip
mv clean_testset_wav/* "$OUTPUT_DIR/test/clean/"
rmdir clean_testset_wav

echo "Extracting noisy testset..."
unzip -q -o noisy_testset_wav.zip
mv noisy_testset_wav/* "$OUTPUT_DIR/test/noisy/"
rmdir noisy_testset_wav

# Create validation set (copy from test or split from train)
echo ""
echo "Creating validation set (using test set)..."
cp "$OUTPUT_DIR/test/clean/"* "$OUTPUT_DIR/valid/clean/"
cp "$OUTPUT_DIR/test/noisy/"* "$OUTPUT_DIR/valid/noisy/"

# Verify
echo ""
echo "=========================================="
echo "Dataset prepared successfully!"
echo "=========================================="
echo ""
echo "Directory structure:"
echo "$OUTPUT_DIR/"
echo "├── train/"
echo "│   ├── clean/  ($(ls -1 "$OUTPUT_DIR/train/clean/" | wc -l) files)"
echo "│   └── noisy/  ($(ls -1 "$OUTPUT_DIR/train/noisy/" | wc -l) files)"
echo "├── valid/"
echo "│   ├── clean/  ($(ls -1 "$OUTPUT_DIR/valid/clean/" | wc -l) files)"
echo "│   └── noisy/  ($(ls -1 "$OUTPUT_DIR/valid/noisy/" | wc -l) files)"
echo "└── test/"
echo "    ├── clean/  ($(ls -1 "$OUTPUT_DIR/test/clean/" | wc -l) files)"
echo "    └── noisy/  ($(ls -1 "$OUTPUT_DIR/test/noisy/" | wc -l) files)"
echo ""
echo "To start training:"
echo "  python train_noise_cond.py --base_dir $OUTPUT_DIR --backbone ncsnpp_v2_cond"

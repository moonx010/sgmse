#!/usr/bin/env python3
"""Check time alignment between clean and enhanced audio."""
import soundfile as sf
import numpy as np
from scipy import signal
from glob import glob

clean_dir = 'data/test_mixtures/voices_reverb/clean'
enhanced_dir = 'results/voices_reverb/wsj0_reverb'

clean_files = sorted(glob(f'{clean_dir}/*.wav'))[:5]

for clean_path in clean_files:
    filename = clean_path.split('/')[-1]
    enhanced_path = f'{enhanced_dir}/{filename}'

    clean, sr = sf.read(clean_path)
    enhanced, _ = sf.read(enhanced_path)

    min_len = min(len(clean), len(enhanced))
    clean = clean[:min_len]
    enhanced = enhanced[:min_len]

    corr = signal.correlate(enhanced, clean, mode='full')
    lag = np.argmax(corr) - len(clean) + 1

    print(f'{filename}: lag = {lag} samples ({lag/sr*1000:.1f} ms)')

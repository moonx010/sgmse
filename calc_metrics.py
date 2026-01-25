import os
from os.path import join
from glob import glob
from argparse import ArgumentParser
from soundfile import read
from tqdm import tqdm
from pesq import pesq
import pandas as pd
import librosa

from pystoi import stoi

from sgmse.util.other import energy_ratios, mean_std


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True, help='Directory containing the clean data')
    parser.add_argument("--noisy_dir", type=str, required=True, help='Directory containing the noisy data')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory containing the enhanced data')
    args = parser.parse_args()

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [],  "si_sar": []}

    # Evaluate standard metrics
    noisy_files = []
    noisy_files += sorted(glob(join(args.noisy_dir, '*.wav')))
    noisy_files += sorted(glob(join(args.noisy_dir, '**', '*.wav')))
    for noisy_file in tqdm(noisy_files):
        filename = noisy_file.replace(args.noisy_dir, "")[1:]

        # Try to find matching clean file
        # First, try same filename (for create_test_mixtures.py output)
        clean_filename = filename
        clean_path = join(args.clean_dir, clean_filename)

        # If not found, try original VoiceBank-DEMAND format (e.g., p232_001_-5dB.wav -> p232_001.wav)
        if not os.path.exists(clean_path) and 'dB' in filename:
            clean_filename = filename.split("_")[0] + ".wav"
            clean_path = join(args.clean_dir, clean_filename)

        x, sr_x = read(clean_path)
        y, sr_y = read(join(args.noisy_dir, filename))
        x_hat, sr_x_hat = read(join(args.enhanced_dir, filename))

        # Resample if sample rates don't match
        target_sr = sr_x
        if sr_y != target_sr:
            y = librosa.resample(y, orig_sr=sr_y, target_sr=target_sr)
        if sr_x_hat != target_sr:
            x_hat = librosa.resample(x_hat, orig_sr=sr_x_hat, target_sr=target_sr)

        # Ensure same length
        min_len = min(len(x), len(y), len(x_hat))
        x, y, x_hat = x[:min_len], y[:min_len], x_hat[:min_len]

        n = y - x 
        x_hat_16k = librosa.resample(x_hat, orig_sr=target_sr, target_sr=16000) if target_sr != 16000 else x_hat
        x_16k = librosa.resample(x, orig_sr=target_sr, target_sr=16000) if target_sr != 16000 else x
        data["filename"].append(filename)
        data["pesq"].append(pesq(16000, x_16k, x_hat_16k, 'wb'))
        data["estoi"].append(stoi(x, x_hat, target_sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])

    # Save results as DataFrame    
    df = pd.DataFrame(data)

    # Print results
    print("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())))
    print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    print("SI-SDR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sdr"].to_numpy())))
    print("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())))
    print("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())))

    # Save average results to file
    log = open(join(args.enhanced_dir, "_avg_results.txt"), "w")
    log.write("PESQ: {:.2f} ± {:.2f}".format(*mean_std(df["pesq"].to_numpy())) + "\n")
    log.write("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())) + "\n")
    log.write("SI-SDR: {:.1f} ± {:.2f}".format(*mean_std(df["si_sdr"].to_numpy())) + "\n")
    log.write("SI-SIR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sir"].to_numpy())) + "\n")
    log.write("SI-SAR: {:.1f} ± {:.1f}".format(*mean_std(df["si_sar"].to_numpy())) + "\n")

    # Save DataFrame as csv file
    df.to_csv(join(args.enhanced_dir, "_results.csv"), index=False)

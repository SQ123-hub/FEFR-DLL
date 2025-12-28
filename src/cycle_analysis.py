import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def standardize_data(series):
    mean_val = np.mean(series)
    std_val = np.std(series)
    if std_val == 0:
        return series - mean_val
    return (series - mean_val) / std_val


def calculate_autocorrelation(series, k):
    n = len(series)
    mean = np.mean(series)
    numerator = np.sum((series[:n - k] - mean) * (series[k:] - mean))
    denominator = np.sum((series - mean) ** 2)
    if denominator == 0: return 0
    return numerator / denominator


def analyze_cycles(csv_path):
    print(f"[*] Analyzing cycles from {csv_path}...")
    df = pd.read_csv(csv_path)
    X = df['Bytes'].values

    X_std = standardize_data(X)

    N = len(X_std)
    T_sample = 1.0

    yf = fft(X_std)
    xf = fftfreq(N, T_sample)[:N // 2]
    amplitudes = 2.0 / N * np.abs(yf[0:N // 2])

    idx_max = np.argmax(amplitudes[1:]) + 1
    freq_T = xf[idx_max]
    cycle_T_duration = 1 / freq_T if freq_T > 0 else 0

    print(f"\n=== Analysis Results ===")
    print(f"Dominant Frequency: {freq_T:.4f} Hz")
    print(f"Estimated Outer Cycle (T): {cycle_T_duration:.2f} seconds (or time units)")

    indices = np.argsort(amplitudes[1:])[::-1][:5] + 1
    print("Top 5 Frequencies components:")
    for idx in indices:
        f = xf[idx]
        period = 1 / f if f > 0 else 0
        print(f" - Freq: {f:.4f}, Period: {period:.2f}")

    if len(indices) > 1:
        cycle_t_duration = 1 / xf[indices[1]]
        print(f"Estimated Inner Cycle (t): {cycle_t_duration:.2f} seconds")
    else:
        cycle_t_duration = cycle_T_duration / 2
        print(f"Estimated Inner Cycle (t): {cycle_t_duration:.2f} seconds (Default: T/2)")

    lag_T = int(cycle_T_duration)
    if lag_T < N:
        rho = calculate_autocorrelation(X_std, lag_T)
        print(f"Autocorrelation for T (k={lag_T}): {rho:.4f} (Should be close to 1)")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(X_std, label='Standardized Traffic')
    plt.title('Network Traffic Time Series (Normalized) [Fig. 2a]')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(xf, amplitudes)
    plt.title('Traffic Spectrum (FFT) [Fig. 2b]')
    plt.xlabel('Frequency')
    plt.ylabel('Normalized Amplitude')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('data/cycle_analysis_result.png')
    print("[*] Plot saved to data/cycle_analysis_result.png")


if __name__ == '__main__':
    analyze_cycles('data/traffic_series.csv')
import os
import glob
import pandas as pd
import numpy as np
import subprocess


def parse_itg_logs(raw_log_dir, output_csv, sampling_interval=1.0):
    print(f"[*] Parsing logs from {raw_log_dir}...")

    log_files = glob.glob(os.path.join(raw_log_dir, 'recv_log_*.log'))

    all_traffic_data = []

    for log_file in log_files:
        txt_file = log_file + '.txt'
        subprocess.run(f"ITGDec {log_file} > {txt_file}", shell=True)

        try:
            df = pd.read_csv(txt_file, sep='\s+', on_bad_lines='skip', header=None)

            if df.empty:
                continue

            subset = df.iloc[:, [6, 7]]
            subset.columns = ['Time', 'Bytes']
            all_traffic_data.append(subset)

        except Exception as e:
            print(f"[!] Error parsing {txt_file}: {e}")
        finally:
            if os.path.exists(txt_file):
                os.remove(txt_file)

    if not all_traffic_data:
        print("[!] No valid traffic data found.")
        return

    full_df = pd.concat(all_traffic_data)

    full_df = full_df.sort_values('Time')

    start_time = full_df['Time'].min()
    full_df['Time'] = full_df['Time'] - start_time

    full_df['TimeBin'] = (full_df['Time'] // sampling_interval).astype(int)
    time_series = full_df.groupby('TimeBin')['Bytes'].sum().reset_index()

    time_series.to_csv(output_csv, index=False)
    print(f"[*] Time series saved to {output_csv}. Total data points: {len(time_series)}")


if __name__ == '__main__':
    parse_itg_logs('data/raw_logs', 'data/traffic_series.csv', sampling_interval=1.0)
import os
import json
import glob
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dirs', type=str, nargs='+')
    parser.add_argument('--result_file', type=str, default='vggt_results.txt')
    parser.add_argument('--start_row', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    metrics_dict = {}

    for output_dir in tqdm(args.output_dirs, desc="Accumulating Results"):
        if not os.path.isdir(output_dir):
            print(f"Warning: {output_dir} is not a directory. Skipping.")
            continue

        with open(os.path.join(output_dir, args.result_file), 'r') as f:
            lines = f.readlines()

        # Extract the line containing the metrics (third line)
        for dataline in lines[args.start_row:]:
            dataline = dataline.strip()
            metric_key = dataline.split(': ')[0]
            metric_val = float(dataline.split(': ')[-1])
            if metric_key not in metrics_dict:
                metrics_dict[metric_key] = []
            metrics_dict[metric_key].append(metric_val)

    for metric_key in metrics_dict:
        metrics_dict[metric_key] = np.mean(metrics_dict[metric_key])

    assert args.save_path.endswith('.txt'), "The save_path should be a txt file."
    with open(args.save_path, 'w') as f:
        f.write(f"Average Metrics of {args.output_dirs}: \n")
        for metric_key in metrics_dict:
            f.write(f"{metric_key}: {metrics_dict[metric_key]}\n")
    print(f"Averaged Metrics of {args.output_dirs}: \n", metrics_dict)



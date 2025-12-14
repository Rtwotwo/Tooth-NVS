import os
import pandas as pd
import numpy as np
import argparse

def gather_results(output_dir, format_float=False, result_filename='results.txt'):
    data = []

    # Walk through subdirectories
    for subdir, _, files in os.walk(output_dir):
        if result_filename in files:
            result_path = os.path.join(subdir, result_filename)
            with open(result_path, 'r') as f:
                contents = f.read()
                # Split on commas and parse key-value pairs
                items = contents.strip().split(',')
                metrics = {}
                for item in items:
                    if ':' in item:
                        key, value = item.split(':', 1)
                        try:
                            metrics[key.strip()] = float(value.strip())
                        except ValueError:
                            metrics[key.strip()] = value.strip()
                metrics['folder'] = os.path.basename(subdir)
                data.append(metrics)
        else:
            print(f"Warning: {result_filename} not found in {subdir}")

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Reorder columns to have 'folder' first
    cols = ['folder'] + [col for col in df.columns if col != 'folder']
    df = df[cols]

    # Compute average and std for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    avg_row = {col: df[col].mean() for col in numeric_cols}
    std_row = {col: df[col].std() for col in numeric_cols}
    avg_row['folder'] = 'AVERAGE'
    std_row['folder'] = 'STD'

    # Sort the main data rows by folder name
    df_sorted = df.sort_values(by='folder', key=lambda col: col.str.lower())

    # Append summary rows
    df_final = pd.concat([df_sorted, pd.DataFrame([avg_row, std_row])], ignore_index=True)
    df_final = df_final[cols]  # Ensure column order

    # Apply float formatting to 4 significant digits if requested
    if format_float:
        df_final[numeric_cols] = df_final[numeric_cols].applymap(lambda x: float(f"{x:.4g}") if pd.notnull(x) else x)

    # Save to CSV in the output_dir
    file_name = result_filename.split('.')[0]
    output_csv = os.path.join(output_dir, f'aggregated_{file_name}.csv')
    df_final.to_csv(output_csv, index=False)
    print(f"Saved aggregated results with summary to {output_csv}")

# Use argparse to pass in output_dir
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate results from results.txt files.')
    parser.add_argument('output_dir', type=str, help='Directory containing subfolders with results.txt files')
    parser.add_argument('--result_filename', type=str, default='results.txt', help='Name of the results file to look for in subfolders')
    parser.add_argument('--format_float', action='store_true', help='Format floating numbers to 4 significant digits')
    args = parser.parse_args()

    gather_results(args.output_dir, format_float=args.format_float, result_filename=args.result_filename)

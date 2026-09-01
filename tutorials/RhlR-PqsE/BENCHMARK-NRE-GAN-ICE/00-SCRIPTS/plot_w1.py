# "plot_w1.py"
import os
import argparse
import glob
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# A distinct color palette for up to 12 tests
COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf', '#393b79', '#9c9ede'
]

def extract_last_w1(filepath):
    """
    Reads the file and extracts w1 from the LAST instance of a line 
    formatted as [w1 w2].
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Iterate backwards through the lines to find the last instance
        for line in reversed(lines):
            line = line.strip()
            # Check if line looks like [w1 w2]
            if line.startswith('[') and line.endswith(']'):
                # Strip the brackets and split by whitespace
                inner_content = line[1:-1].strip()
                parts = inner_content.split()
                
                if len(parts) >= 2:
                    try:
                        w1 = float(parts[0])
                        return w1
                    except ValueError:
                        continue # If they aren't numbers, keep looking backwards
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot w1 over time for different REP values and save as PNG.")
    parser.add_argument("directory", help="Path to the main directory containing REP-YYY subdirectories")
    args = parser.parse_args()

    main_dir = args.directory

    if not os.path.isdir(main_dir):
        print(f"Error: {main_dir} is not a valid directory.")
        return

    # Get the absolute path to properly extract the name of the main directory
    abs_main_dir = os.path.abspath(main_dir)
    main_dir_name = os.path.basename(abs_main_dir)

    # Find all REP-YYY directories
    beta_dirs = glob.glob(os.path.join(abs_main_dir, 'REP-*'))

    if not beta_dirs:
        print("No directories matching 'REP-*' found.")
        return

    # Sort directories by the YYY value so the legend is ordered logically
    def get_beta_val(dir_path):
        basename = os.path.basename(dir_path)
        try:
            return float(basename.split('REP-')[1])
        except ValueError:
            return 0.0

    beta_dirs.sort(key=get_beta_val)

    plt.figure(figsize=(10, 6))

    # Store w1 values grouped by epoch across all tests
    w1_by_time = defaultdict(list)

    # Loop through each REP directory
    for ii, b_dir in enumerate(beta_dirs):
        subdir_name = os.path.basename(b_dir)
        color = COLORS[ii % len(COLORS)]  # Prevents IndexError if >12 directories
        
        # Find all log.inference-small* files
        log_files = glob.glob(os.path.join(b_dir, 'log.inference-small*'))
        
        data_points = []
        
        for filepath in log_files:
            filename = os.path.basename(filepath)
            
            # Determine the time (XXX)
            if filename == 'log.inference-small':
                time = 100
            else:
                time_str = filename.split('.')[-1]
                if time_str.isdigit():
                    time = int(time_str)
                else:
                    continue # Skip files that don't match the exact pattern
            
            # Extract w1
            w1 = extract_last_w1(filepath)
            if w1 is not None:
                data_points.append((time, w1))
                w1_by_time[time].append(w1)
                
        # Sort data points by time so the line plot connects properly
        data_points.sort(key=lambda x: x[0])

        if data_points:
            # Unzip into two lists: times and w1s
            times, w1s = zip(*data_points)
            
            # Plot with line and filled circles
            plt.plot(times, w1s, marker='o', linestyle='-', color=color, label=subdir_name)
            
    # Plot the average across all tests as a bold black line
    if w1_by_time:
        avg_times = sorted(w1_by_time.keys())
        avg_w1s = [sum(w1_by_time[t]) / len(w1_by_time[t]) for t in avg_times]
        plt.plot(avg_times, avg_w1s, color='black', linestyle='-', linewidth=2.5,
                 label='Average', zorder=10)

    # Formatting the plot
    plt.title(main_dir_name, fontsize=14, fontweight='bold')
    plt.xlabel("Time (# epochs)")
    plt.ylabel("w1")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Place the legend slightly outside the plot so it doesn't cover data
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="Subdirectories") 
    
    # Ensure layout fits everything perfectly
    plt.tight_layout() 
    
    # Save to PNG
    output_filename = f"{main_dir_name}_w1_plot.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Success! Plot saved as: {output_filename}")

if __name__ == "__main__":
    main()

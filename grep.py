import sys
from collections import deque
import os
import pandas as pd
import argparse
import glob

def calculate_success_rates(log_content):
    stats = {}
    lines = log_content.strip().split('\n')
    i = 0
    while i < len(lines) - 1:
        line1 = lines[i].strip()
        line2 = lines[i+1].strip()

        # Check if lines match the expected pattern
        if ':' in line1 and 'Average success' in line2:
            try:
                # Extract operation name (after first ':')  
                op_name = line1.split(':', 1)[1].strip()
                # Extract success rate (between 'Average success' and '<<<')
                rate_str = line2.split('Average success')[1].split('<<<')[0].strip()
                rate = float(rate_str)

                # Store the rate
                if op_name not in stats:
                    stats[op_name] = []
                stats[op_name].append(rate)

                # Skip the next line since we've processed it as the success rate
                i += 1
            except (ValueError, IndexError):
                # Ignore pairs that don't parse correctly (e.g., non-float rate)
                pass

        i += 1 # Move to the next potential operation line

    # Calculate averages
    averages = {}
    for op, rates in stats.items():
        averages[op] = sum(rates) / len(rates) if rates else 0.0 # Avoid division by zero
    return averages

def grep():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        return 1
    
    file_path = sys.argv[1]
    pattern = "  Average success"  # Two spaces followed by "Average success"
    log_file = "grep.log"  # Output log file
    
    try:
        # Use a deque to keep track of recent lines
        buffer = deque(maxlen=5)  # Keep last 5 lines to handle both cases
        
        # Open the output file
        with open(log_file, 'w') as out_file:
            # Open the input file
            with open(file_path, 'r') as in_file:
                line_number = 0
                
                for line in in_file:
                    line_number += 1
                    buffer.append((line_number, line.rstrip()))
                    
                    # Check if the current line matches the pattern
                    if pattern in line:
                        # Check if the buffer contains "Saving video to"
                        has_saving_video = any("Saving video to" in item[1] for item in buffer)
                        
                        if has_saving_video:
                            # Case 1: Has "Saving video to"
                            # Print first line (buffer[0])
                            if len(buffer) >= 5:
                                first_line = buffer[0]
                                out_file.write(f"{first_line[0]}: {first_line[1]}\n")
                        else:
                            # Case 2: No "Saving video to"
                            # Print first line (buffer[0])
                            if len(buffer) >= 4:
                                first_line = buffer[1]
                                out_file.write(f"{first_line[0]}: {first_line[1]}\n")
                        
                        # Print current matching line in both cases
                        out_file.write(f"{line_number}: {line.rstrip()} <<< MATCH\n")
                        out_file.write("\n")  # Add an empty line for better readability
        
        return 0
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 1
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1


def process_csv_file(input_file, bridge_file_name="bridge.csv", frac_file_name="frac.csv"):
    """
    Reads a CSV file from the specified path and processes it to create two outputs:
    1. Column 16 onwards excluding last 3 columns (bridge.csv)
    2. Selected columns from first 16 columns in specific order (frac.csv):
       - First three columns: df_frac[0, 9, 10]
       - Last three columns: df_frac[1, 8, 11]
    
    Args:
        input_file (str): Path to the input CSV file
        bridge_file_name (str): Name of the output CSV for columns 16+ (minus last 3)
        frac_file_name (str): Name of the output CSV for selected columns in specific order
        
    Returns:
        tuple: (df_frac_combined, df_bridge) The processed DataFrames written to output files
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file not found at {input_file}")
            return None, None
            
        # Get the directory path from the input file
        input_dir = os.path.dirname(input_file)
            
        # Create the output file paths
        bridge_file = os.path.join(input_dir, bridge_file_name)
        frac_file = os.path.join(input_dir, frac_file_name)
            
        # Read the CSV file
        df = pd.read_csv(input_file)
            
        # Make sure we have enough columns
        if df.shape[1] <= 16:
            print("Warning: The DataFrame has 16 or fewer columns. Cannot split as requested.")
            return None, None
                
        # Get the first 16 columns
        df_frac = df.iloc[:, :16]
        
        # Select columns in the specified order
        # First three columns: [0, 9, 10], Last three columns: [1, 8, 11]
        df_frac_combined = df_frac.iloc[:, [0, 9, 10, 1, 8, 11]].copy()
        
        # Add average columns for frac data
        if df_frac_combined.shape[1] >= 6:  # Make sure we have enough columns
            # matching average (columns 0, 1, 2)
            df_frac_combined['matching_average'] = df_frac_combined.iloc[:, [0, 1, 2]].mean(axis=1).round(3)
            # variant average (columns 3, 4, 5)
            df_frac_combined['variant_average'] = df_frac_combined.iloc[:, [3, 4, 5]].mean(axis=1).round(3)
            # overall average (all 6 columns)
            df_frac_combined['overall_average'] = df_frac_combined.iloc[:, :6].mean(axis=1).round(3)
        
        # Get bridge data (columns 16 onwards, excluding last 3)
        df_bridge = df.iloc[:, 16:]
        if df_bridge.shape[1] > 3:
            df_bridge = df_bridge.iloc[:, :-3]  # Exclude the last 3 columns
        
        # Add average column for matching_entire (columns 1, 3, 5, 7)
        if df_bridge.shape[1] >= 8:  # Make sure we have enough columns
            matching_entire_cols = [1, 3, 5, 7]  # matching_entire columns
            df_bridge['matching_entire_average'] = df_bridge.iloc[:, matching_entire_cols].mean(axis=1).round(3)
            
        # Write the processed DataFrames to the new CSV files
        df_frac_combined.to_csv(frac_file, index=False) 
        df_bridge.to_csv(bridge_file, index=False)
            
        return df_frac_combined, df_bridge
        
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}")
        return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CSV files by removing specific columns.')
    parser.add_argument('--prefix', type=str, default='./results/', 
                        help='Path prefix containing directories to process')
    
    args = parser.parse_args()
    
    # 创建列表来存储所有的 df_frac_combined 和 df_bridge
    all_frac_dfs = []
    all_bridge_dfs = []
    
    # Find all directories under the prefix
    directories = [d for d in glob.glob(os.path.join(args.prefix, '*')) if os.path.isdir(d)]
    
    for directory in directories:
        # Construct the path to the results.csv file in each directory
        input_file = os.path.join(directory, 'results.csv')
        if os.path.exists(input_file):
            df_frac_combined, df_bridge = process_csv_file(input_file=input_file)
            
            if df_frac_combined is not None:
                df_frac_combined['source_directory'] = os.path.basename(directory)
                all_frac_dfs.append(df_frac_combined)
            
            if df_bridge is not None:
                df_bridge['source_directory'] = os.path.basename(directory)
                all_bridge_dfs.append(df_bridge)
        else:
            print(f"Warning: No results.csv found in {directory}")
    
    if all_frac_dfs:
        combined_frac_df = pd.concat(all_frac_dfs, ignore_index=True)
        
        # Add average columns for combined frac data
        if combined_frac_df.shape[1] >= 6:
            if 'matching_average' not in combined_frac_df.columns:
                # matching average (columns 0, 1, 2)
                combined_frac_df['matching_average'] = combined_frac_df.iloc[:, [0, 1, 2]].mean(axis=1).round(3)
            if 'variant_average' not in combined_frac_df.columns:
                # variant average (columns 3, 4, 5)
                combined_frac_df['variant_average'] = combined_frac_df.iloc[:, [3, 4, 5]].mean(axis=1).round(3)
            if 'overall_average' not in combined_frac_df.columns:
                # overall average (all 6 columns)
                combined_frac_df['overall_average'] = combined_frac_df.iloc[:, :6].mean(axis=1).round(3)
        
        # 按 overall_average 从高到低排序
        if 'overall_average' in combined_frac_df.columns:
            combined_frac_df = combined_frac_df.sort_values('overall_average', ascending=False)
        
        combined_frac_output_file = os.path.join(args.prefix, 'combined_frac.csv')
        combined_frac_df.to_csv(combined_frac_output_file, index=False)
        print(f"Successfully created combined frac CSV file at {combined_frac_output_file}")
    else:
        print("No valid frac data found to combine.")

    if all_bridge_dfs:
        combined_bridge_df = pd.concat(all_bridge_dfs, ignore_index=True)
        
        # Add average column for matching_entire (columns 1, 3, 5, 7) for combined data
        if combined_bridge_df.shape[1] >= 8 and 'matching_entire_average' not in combined_bridge_df.columns:
            matching_entire_cols = [1, 3, 5, 7]  # matching_entire columns
            combined_bridge_df['matching_entire_average'] = combined_bridge_df.iloc[:, matching_entire_cols].mean(axis=1).round(3)
        
        # 按 matching_entire_average 从高到低排序
        if 'matching_entire_average' in combined_bridge_df.columns:
            combined_bridge_df = combined_bridge_df.sort_values('matching_entire_average', ascending=False)
        
        # 创建 bridge 总表的输出路径
        combined_bridge_output_file = os.path.join(args.prefix, 'combined_bridge.csv')
        
        # 保存 bridge 总表
        combined_bridge_df.to_csv(combined_bridge_output_file, index=False)
        print(f"Successfully created combined bridge CSV file at {combined_bridge_output_file}")
    else:
        print("No valid bridge data found to combine.")
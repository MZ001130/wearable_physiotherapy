
import pandas as pd
import glob
import os

def verify_frequency(root_dir='data/dev_raw_data'):
    """
    Calculates and prints the sampling frequency for each sensor file in the specified directory.
    Frequency is calculated as (number of samples) / (last 'seconds_elapsed' value).
    """
    print(f"Analyzing sensor files in: {root_dir}\\n")
    
    # Find all Accelerometer.csv and Gyroscope.csv files recursively
    sensor_files = glob.glob(os.path.join(root_dir, '**', 'Accelerometer.csv'), recursive=True)
    sensor_files.extend(glob.glob(os.path.join(root_dir, '**', 'Gyroscope.csv'), recursive=True))

    if not sensor_files:
        print("No sensor files found to analyze.")
        return

    # A list to hold results for a summary at the end
    results = []

    for file_path in sorted(sensor_files):
        try:
            df = pd.read_csv(file_path)
            
            # Ensure the dataframe is not empty and the required column exists
            if df.empty or 'seconds_elapsed' not in df.columns:
                print(f"Skipping {file_path}: File is empty or missing 'seconds_elapsed' column.")
                continue

            num_samples = len(df)
            last_timestamp = df['seconds_elapsed'].iloc[-1]

            # Avoid division by zero
            if last_timestamp > 0:
                frequency = num_samples / last_timestamp
            else:
                frequency = 0

            # Store result
            results.append({'file': os.path.relpath(file_path), 'frequency': frequency, 'samples': num_samples, 'duration_s': last_timestamp})
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if not results:
        print("\nNo data was successfully processed.")
        return

    # Print results in a structured way
    print(f"{'File':<60} | {'Frequency (Hz)':>15} | {'Samples':>10} | {'Duration (s)':>12}")
    print(f"{'-'*60} | {'-'*15} | {'-'*10} | {'-'*12}")
    for res in results:
        print(f"{res['file']:<60} | {res['frequency']:>15.2f} | {res['samples']:>10} | {res['duration_s']:>12.2f}")
    
    # Calculate and print average frequency
    avg_freq = sum(r['frequency'] for r in results) / len(results)
    print("\n" + "="*100)
    print(f"Average calculated frequency across all files: {avg_freq:.2f} Hz")
    print("="*100)


if __name__ == '__main__':
    verify_frequency()


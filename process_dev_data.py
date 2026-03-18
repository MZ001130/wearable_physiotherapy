
import pandas as pd
import glob
import re
import os

def process_dev_data(root_dir='data/dev_raw_data', output_file='data/processed/dev_exercise_data.csv'):
    """
    Combines accelerometer and gyroscope data from subfolders into a single CSV file.

    Args:
        root_dir (str): The root directory containing the raw data folders.
        output_file (str): The path to save the combined CSV file.
    """
    all_data = []
    
    # Use glob to find all subdirectories
    subfolders = glob.glob(os.path.join(root_dir, 'SN_*'))

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        
        # Parse folder name for subject, exercise, and arm
        match = re.match(r'SN_(E\d+)_(L|R)', folder_name)
        if not match:
            print(f"Skipping folder with unexpected name format: {folder_name}")
            continue
            
        exercise = match.group(1)
        arm_char = match.group(2)
        arm = 'left' if arm_char == 'L' else 'right'
        subject = 'sn'

        accel_path = os.path.join(folder, 'Accelerometer.csv')
        gyro_path = os.path.join(folder, 'Gyroscope.csv')

        if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
            print(f"Skipping folder {folder_name}: missing sensor files.")
            continue

        try:
            # Read accelerometer data
            accel_df = pd.read_csv(accel_path)
            accel_df = accel_df.rename(columns={'x': 'ax', 'y': 'ay', 'z': 'az'})
            
            # Read gyroscope data
            gyro_df = pd.read_csv(gyro_path)
            gyro_df = gyro_df.rename(columns={'x': 'wx', 'y': 'wy', 'z': 'wz'})

            # Merge data on 'seconds_elapsed'
            # To handle potential floating point inaccuracies, we can round the merge key
            accel_df['merge_key'] = accel_df['seconds_elapsed'].round(5)
            gyro_df['merge_key'] = gyro_df['seconds_elapsed'].round(5)
            
            merged_df = pd.merge(accel_df[['merge_key', 'ax', 'ay', 'az']], 
                                 gyro_df[['merge_key', 'wx', 'wy', 'wz', 'seconds_elapsed']], 
                                 on='merge_key',
                                 how='inner') # Use inner join to keep only matching timestamps

            # Add metadata columns
            merged_df['subject'] = subject
            merged_df['exercise'] = exercise
            merged_df['arm'] = arm
            
            # Select and reorder columns
            final_cols = ['ax', 'ay', 'az', 'wx', 'wy', 'wz', 'seconds_elapsed', 'subject', 'exercise', 'arm']
            merged_df = merged_df[final_cols]

            all_data.append(merged_df)

        except Exception as e:
            print(f"Error processing folder {folder_name}: {e}")

    if not all_data:
        print("No data processed. Exiting.")
        return

    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Successfully created combined file at: {output_file}")
    print(f"Total rows: {len(combined_df)}")

if __name__ == '__main__':
    process_dev_data()

import pandas as pd
import glob
import os

def load_and_process_data():
    raw_data_path = 'data/raw/*.csv'
    processed_data_path = 'data/processed/'
    output_filename = 'exercise_data.csv'

    # Ensure processed directory exists
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)

    all_files = glob.glob(raw_data_path)
    combined_list = []

    for filename in all_files:
        
        df = pd.read_csv(filename)
        
        base_name = os.path.basename(filename).replace('.csv', '')
        parts = base_name.split('_')
        
        if 'subject_id' not in df.columns:
            df['subject_id'] = parts[0]
        if 'exercise' not in df.columns:
            df['exercise'] = parts[1] if len(parts) > 1 else 'unknown'

        combined_list.append(df)

    full_df = pd.concat(combined_list, ignore_index=True)

    columns_to_keep = ["ax", "ay", "az", "wx", "wy", "wz", "exercise", "subject_id"]
    full_df = full_df[columns_to_keep]

    full_df.to_csv(os.path.join(processed_data_path, output_filename), index=False)
    
    return full_df

if __name__ == "__main__":
    load_and_process_data()

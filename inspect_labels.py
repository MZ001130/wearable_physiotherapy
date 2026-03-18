
import joblib
import numpy as np

def inspect_training_labels(file_path="data/feature/windowed_features.joblib"):
    """
    Loads the feature file and prints the unique exercise labels.
    """
    try:
        data = joblib.load(file_path)
        if 'y' in data:
            y = data['y']
            unique_labels = np.unique(y)
            print("Unique exercise labels found in the original training feature file:")
            print(sorted(unique_labels))
        else:
            print("Error: 'y' (labels) not found in the joblib file.")
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect_training_labels()

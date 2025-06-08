import pandas as pd
import random
import argparse

### USAGE 
#  python select_audios.py /home/spandan/projects/pbl2_audio_watermarking/metadata.csv --hours 100 100_hours_additional_metadata.csv

def select_audios_by_duration(metadata_path, target_duration, is_hours, output_path):
    """
    Randomly select audio files until they reach the target duration.
    
    Args:
        metadata_path (str): Path to the input metadata CSV file
        target_duration (float): Target duration in hours or seconds
        is_hours (bool): If True, target_duration is in hours; if False, in seconds
        output_path (str): Path to save the new metadata CSV file
    """
    # Convert to seconds if input is in hours
    if is_hours:
        target_seconds = target_duration * 3600
        original_unit = "hours"
    else:
        target_seconds = target_duration
        original_unit = "seconds"
    
    # Read the metadata file
    df = pd.read_csv(metadata_path)
    
    # Check if duration column exists
    if 'duration' not in df.columns:
        raise ValueError("The metadata file must contain a 'duration' column")
    
    # Create a copy of the dataframe with all rows
    all_files = df.copy()
    
    # Shuffle the dataframe
    all_files = all_files.sample(frac=1, random_state=random.randint(1, 1000))
    
    # Initialize variables
    selected_files = []
    total_duration = 0
    
    # Select files until we reach the target duration
    for _, row in all_files.iterrows():
        if total_duration >= target_seconds:
            break
        
        selected_files.append(row)
        total_duration += row['duration']  # Duration is already in seconds
    
    # Create a new dataframe with the selected files
    selected_df = pd.DataFrame(selected_files)
    
    # Save the new metadata file
    selected_df.to_csv(output_path, index=False)
    
    # Print a summary
    hours_selected = total_duration / 3600
    print(f"Selected {len(selected_files)} files with a total duration of {hours_selected:.2f} hours ({total_duration:.2f} seconds)")
    print(f"Target duration was {target_duration:.2f} {original_unit} ({target_seconds:.2f} seconds)")
    print(f"New metadata saved to {output_path}")

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Select audio files by duration")
    parser.add_argument("metadata_path", help="Path to the input metadata CSV file")
    
    # Create a mutually exclusive group for duration specification
    duration_group = parser.add_mutually_exclusive_group(required=True)
    duration_group.add_argument("--hours", "-hr", type=float, help="Target duration in hours")
    duration_group.add_argument("--seconds", "-sec", type=float, help="Target duration in seconds")
    
    parser.add_argument("output_path", help="Path to save the new metadata CSV file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Determine if hours or seconds were specified
    if args.hours is not None:
        target_duration = args.hours
        is_hours = True
    else:
        target_duration = args.seconds
        is_hours = False
    
    # Call the function
    select_audios_by_duration(args.metadata_path, target_duration, is_hours, args.output_path)
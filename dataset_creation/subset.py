import os
import csv
import soundfile as sf

def get_duration(filepath):
    try:
        info = sf.info(filepath)
        duration = info.frames / info.samplerate
        return duration
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0.0

def generate_metadata(root_dir, output_csv="metadata.csv"):
    metadata = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".ogg"):
                filepath = os.path.join(root, file)
                duration = get_duration(filepath)
                if duration > 0:
                    metadata.append({"filepath": filepath, "duration": duration})
    
    if metadata:
        keys = metadata[0].keys()
        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metadata)
        print(f"Metadata written to {output_csv}. Total files processed: {len(metadata)}")
    else:
        print("No valid audio files found.")
    
    return metadata

if __name__ == "__main__":
    root_directory = "data/raw_audios/en"
    metadata = generate_metadata(root_directory, output_csv="metadata.csv")

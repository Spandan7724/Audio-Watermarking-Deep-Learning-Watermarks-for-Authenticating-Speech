import os

def delete_zone_identifier_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(":Zone.Identifier"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

if __name__ == "__main__":
    # Get the directory where the script is being run
    current_directory = os.getcwd()
    print(f"Cleaning Zone.Identifier files in: {current_directory}")
    delete_zone_identifier_files(current_directory)

import os

def merge_txt_files(source_dirs, destination_dir):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Loop through each source directory
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"Source directory {source_dir} does not exist. Skipping...")
            continue

        # Walk through the source directory and find all .txt files
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".txt"):
                    # Construct full file path
                    source_file_path = os.path.join(root, file)

                    # Define the destination path for the file
                    dest_file_path = os.path.join(destination_dir, file)

                    # If a file with the same name exists, append a number to the filename to avoid conflicts
                    if os.path.exists(dest_file_path):
                        base_name, extension = os.path.splitext(file)
                        counter = 1
                        while os.path.exists(dest_file_path):
                            new_file_name = f"{base_name}_{counter}{extension}"
                            dest_file_path = os.path.join(destination_dir, new_file_name)
                            counter += 1

                    # Copy the file to the destination directory using os.system and cp command
                    os.system(f"cp '{source_file_path}' '{dest_file_path}'")
                    print(f"Copied {source_file_path} to {dest_file_path}")

    print(f"\nAll .txt files have been merged into {destination_dir}")

if __name__ == "__main__":
    # Example source directories (replace these with your actual directories)
    source_dirs = [
        "/Users/kevincowan/training_data/training_data_3",
        "/Users/kevincowan/training_data/training_data_4",
    ]

    # Destination directory to merge the .txt files into
    destination_dir = "/Users/kevincowan/training_data/fine_tuning_data"

    merge_txt_files(source_dirs, destination_dir)
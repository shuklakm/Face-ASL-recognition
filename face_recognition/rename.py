import os

def rename_files_in_directory(directory_path):
    """
    Rename all files in the given directory to file_1.ext, file_2.ext, ...
    """
    # List all files in the directory
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Loop through each file and rename
    for index, filename in enumerate(files, start=1):
        # Extract file extension
        file_extension = os.path.splitext(filename)[1]
        # Construct new filename
        new_filename = f"Other_{index}{file_extension}"
        # Rename file
        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))
        print(f"Renamed {filename} to {new_filename}")

# Provide the path to your directory here
directory_path = "./data/1"
rename_files_in_directory(directory_path)

import glob
import os

# To clean the current directory:
# files_to_delete = glob.glob("*.txt.txt")

# Or to clean a specific batch directory, use:
files_to_delete = glob.glob("blocaibatch1/*.txt.txt")

for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except OSError as e:
        print(f"Error deleting {file_path}: {e}")
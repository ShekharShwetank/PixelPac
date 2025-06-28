import os
import subprocess

root_for_models = "models/"
list_of_model_dirs = sorted(os.listdir(root_for_models))  # Sort the directories

for folder in list_of_model_dirs:
    folder_path = os.path.join(root_for_models, folder)
    
    # Skip if not a directory
    if not os.path.isdir(folder_path):
        continue

    py_file = os.path.join(folder_path, f"{folder}.py")
    
    # Check if the Python file exists in the folder
    if os.path.isfile(py_file):
        print(f"Running: {py_file}")
        subprocess.run(["python3", py_file])  # or use "python" depending on your system
    else:
        print(f"Skipped: {folder}, file not found => {py_file}")
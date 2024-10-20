import os
import argparse

# Set up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('folder_path', help='Path to the folder containing PDF files')
args = parser.parse_args()

# Walk through the directory tree
for root, _, files in os.walk(args.folder_path):
    for file in files:
        # Process only PDF files
        if file.endswith('.pdf'):
            # Construct full file path
            file_path = os.path.join(root, file)
            
            # Create output folder
            output_folder = os.path.join(root, 'output')
            os.makedirs(output_folder, exist_ok=True)
            
            # Run marker_single command
            os.system(f'marker_single {file_path} {output_folder} --batch_multiplier 2')
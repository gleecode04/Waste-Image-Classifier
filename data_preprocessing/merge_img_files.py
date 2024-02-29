import os
import shutil

# Source directory containing subdirectories with images
source_dir = 'C:\\Users\\82109\\Desktop\\Deep Learning Project\\img\\green-glass'

# Destination directory where all images will be moved
destination_dir = 'C:\\Users\\82109\\Desktop\\Deep Learning Project\\img\\glass'

# Walk through the source directory and its subdirectories
for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            source_file_path = os.path.join(root, file)
            # Construct destination file path
            destination_file_path = os.path.join(destination_dir, file)
            # Check if the file already exists in the destination directory
                # Move the file to the destination directory
            shutil.move(source_file_path, destination_file_path)
           

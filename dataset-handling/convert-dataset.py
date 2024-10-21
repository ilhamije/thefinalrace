import os
import shutil

# Define paths for the old and new structures
source_root = "RescueNet"
destination_root = "RescueNet-Converted"

# Define target directories for images and annotations
img_dir = os.path.join(destination_root, "img_dir")
ann_dir = os.path.join(destination_root, "ann_dir")

# Ensure the new directories exist
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(img_dir, split), exist_ok=True)
    os.makedirs(os.path.join(ann_dir, split), exist_ok=True)


def move_files(source_subdir, target_subdir, is_annotation):
    """Move files from the old structure to the new structure."""
    for root, _, files in os.walk(source_subdir):
        for file in files:
            if (is_annotation and file.endswith('_lab.png')) or (not is_annotation and file.endswith('.jpg')):
                split_type = os.path.basename(os.path.dirname(root)).split(
                    '-')[0]  # Extract 'train', 'val', or 'test'
                target_path = os.path.join(target_subdir, split_type, file)
                source_path = os.path.join(root, file)
                # Use copy2 to retain metadata
                shutil.copy2(source_path, target_path)
                print(f"Moved {file} to {target_path}")


# Move the original images to the img_dir
move_files(os.path.join(source_root, "train", "train-org-img"),
           img_dir, is_annotation=False)
move_files(os.path.join(source_root, "val", "val-org-img"),
           img_dir, is_annotation=False)
move_files(os.path.join(source_root, "test", "test-org-img"),
           img_dir, is_annotation=False)

# Move the annotation images to the ann_dir
move_files(os.path.join(source_root, "train", "train-label-img"),
           ann_dir, is_annotation=True)
move_files(os.path.join(source_root, "val", "val-label-img"),
           ann_dir, is_annotation=True)
move_files(os.path.join(source_root, "test", "test-label-img"),
           ann_dir, is_annotation=True)

print("Folder structure conversion completed.")

import os
import cv2


def resize_with_aspect_ratio(image, target_height=512):
    """Resize an image to maintain a 4:3 aspect ratio with the given target height."""
    h, w = image.shape[:2]

    # Calculate the width for the 4:3 aspect ratio
    target_width = int((4 / 3) * target_height)

    # Resize the image
    resized = cv2.resize(image, (target_width, target_height),
                         interpolation=cv2.INTER_AREA)
    return resized


def resize_image(input_path, output_path, target_height=512):
    """Resize an image to the specified size (maintaining aspect ratio) and save it."""
    try:
        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Skipping {input_path}: not a valid image.")
            return

        # Resize with aspect ratio 4:3 and the specified height
        resized = resize_with_aspect_ratio(image, target_height)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save the resized image
        cv2.imwrite(output_path, resized)
        print(f"Resized and saved: {output_path}")
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")


def process_directory(input_dir, output_dir, target_height=512):
    """Recursively process all images in the directory."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, relative_path)
                resize_image(input_path, output_path, target_height)


if __name__ == "__main__":
    # Set input directory, output directory, and target height
    input_directory = "/root/folder-docker/occlusion-aware/thefinalrace/data/ori-rescuenet"
    output_directory = "/root/folder-docker/occlusion-aware/thefinalrace/data/rescuenet"
    target_height = 512  # Height to maintain with 4:3 ratio

    process_directory(input_directory, output_directory, target_height)
    print("Resizing process completed.")

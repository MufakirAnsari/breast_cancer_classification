import os

def load_image_paths(base_directory):
    """
    Load all image file paths from the base directory.
    """
    breast_img = []
    subdirectories = [f.path for f in os.scandir(base_directory) if f.is_dir()]
    for subdirectory in subdirectories:
        for root, dirs, files in os.walk(subdirectory):
            for file in files:
                if file.lower().endswith(".png"):  # Case-insensitive check for .png files
                    breast_img.append(os.path.join(root, file))
    return breast_img


def split_images_by_class(image_paths):
    """
    Split images into cancerous and non-cancerous categories based on folder names.
    """
    non_cancer_images = [img for img in image_paths if "\\0\\" in img]
    cancer_images = [img for img in image_paths if "\\1\\" in img]
    return non_cancer_images, cancer_images
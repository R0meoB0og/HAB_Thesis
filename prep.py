# Romain Huet -- Script to process the images and export as PNG.

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import splitfolders as sf
import shutil

# folder paths
image_string = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\zeeland_data\\images\\"
label_string = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\zeeland_data\\labels\\"

# file paths
image_paths = sorted(os.listdir(image_string))
label_paths = sorted(os.listdir(label_string))

# ensure that folders have same length
assert (len(image_paths) == len(label_paths))

# Create binary mask function
def create_binary_mask(mask_image):
    hsv_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2HSV)
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 20, 255], dtype=np.uint8)
    black_lower = np.array([0, 0, 0], dtype=np.uint8)
    black_upper = np.array([180, 255, 50], dtype=np.uint8)
    grey_lower = np.array([0, 0, 50], dtype=np.uint8)
    grey_upper = np.array([180, 50, 200], dtype=np.uint8)
    white_mask = cv2.inRange(hsv_mask, white_lower, white_upper)
    black_mask = cv2.inRange(hsv_mask, black_lower, black_upper)
    grey_mask = cv2.inRange(hsv_mask, grey_lower, grey_upper)
    binary_mask = np.zeros_like(white_mask)
    binary_mask[black_mask == 255] = 0
    non_algae_mask = cv2.bitwise_or(white_mask, black_mask)
    non_algae_mask = cv2.bitwise_or(non_algae_mask, grey_mask)
    algae_mask = cv2.bitwise_not(non_algae_mask)
    binary_mask[algae_mask == 255] = 1
    return binary_mask


# Load images and labels
images_labels_pairs = []
for img_path, lbl_path in zip(image_paths, label_paths):
    image = Image.open(image_string + img_path)
    image = image.convert("RGB")
    image_array = np.array(image)

    mask_image = cv2.bitwise_not(cv2.imread(label_string + lbl_path, cv2.IMREAD_COLOR)) * 255
    binary_mask = create_binary_mask(mask_image) * 255
    binary_mask = np.where(binary_mask == 255, 1, binary_mask)

    images_labels_pairs.append((image_array, binary_mask))


# Tiling
def tiles(pairs, tile_size):
    tiled_pairs = []
    for image, label in pairs:
        num_tiles_x = image.shape[1] // tile_size
        num_tiles_y = image.shape[0] // tile_size
        for tile_y in range(num_tiles_y):
            for tile_x in range(num_tiles_x):
                start_x = tile_x * tile_size
                start_y = tile_y * tile_size
                image_tile = image[start_y:start_y + tile_size, start_x:start_x + tile_size]
                label_tile = label[start_y:start_y + tile_size, start_x:start_x + tile_size]
                if image_tile.shape == (tile_size, tile_size, 3) and label_tile.shape == (tile_size, tile_size):
                    tiled_pairs.append((image_tile, label_tile))
    return tiled_pairs


tile_size = 100
tiled_pairs = tiles(images_labels_pairs, tile_size)


# Filter tiles with at least 20 algae pixels
def filter_tiles(pairs, min_ship_pixels):
    filtered_pairs = []
    for image, label in pairs:
        if np.sum(label == 1) >= min_ship_pixels:
            filtered_pairs.append((image, label))
    return filtered_pairs

filtered_pairs = filter_tiles(tiled_pairs, 20)

''' SKIP WHEN TESTING NEW LAKES
# Augmentation
def augment_pairs(pairs, seed=36):
    augmented_pairs = []

    # Create a random generator with the specified seed
    rng = tf.random.Generator.from_seed(seed)

    for image, label in pairs:
        # Generate a random seed for the current pair to ensure same transformation
        curr_seed = rng.make_seeds(2)[0]

        # Apply the same transformations to both image and label using the current seed
        augmented_image = tf.image.stateless_random_flip_left_right(image, seed=curr_seed).numpy()
        augmented_image = tf.image.stateless_random_flip_up_down(augmented_image, seed=curr_seed).numpy()

        label = np.expand_dims(label, axis=-1)  # Add channel dimension
        augmented_label = tf.image.stateless_random_flip_left_right(label, seed=curr_seed).numpy()
        augmented_label = tf.image.stateless_random_flip_up_down(augmented_label, seed=curr_seed).numpy()
        augmented_label = np.squeeze(augmented_label, axis=-1)  # Remove the added channel dimension

        augmented_pairs.append((augmented_image, augmented_label))

    return augmented_pairs

# Assume `filtered_pairs` is defined
augmented_pairs = augment_pairs(filtered_pairs)
all_pairs = filtered_pairs + augmented_pairs
'''

all_pairs = images_labels_pairs

# Save images and labels
image_dir = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\zeeland\\images\\"
label_dir = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\zeeland\\labels\\"

for i, (image, label) in enumerate(all_pairs):
    image_name = f"image_{i}.png"
    label_name = f"label_{i}.png"
    Image.fromarray(image).save(os.path.join(image_dir, image_name))
    label = (label).astype(np.uint8)  # convert back to 0-255 range
    Image.fromarray(label, mode='L').save(os.path.join(label_dir, label_name))

r''' skip when testing new lake
# Split test and training datasets
data_folder = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\green_bay"
output_folder = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\green_bay_split"
sf.ratio(data_folder, output=output_folder, seed=1337, ratio=(.75, .25))
'''

### ARCHIVE ###
r''' 
def shorten_name(name):
    parts = name.split('.')
    if len(parts) > 2:
        return '.'.join(parts[:2])
    return name

def find_unmatched_pairs(image_path, label_path):
    # Get sorted lists of image and label names
    image_names = sorted(os.listdir(image_path))
    label_names = sorted(os.listdir(label_path))

    # Strip file extensions and shorten names for comparison
    image_names_stripped = [shorten_name(os.path.splitext(name)[0]) for name in image_names]
    label_names_stripped = [shorten_name(os.path.splitext(name)[0]) for name in label_names]

    # Find images without corresponding labels
    unmatched_images = set(image_names_stripped) - set(label_names_stripped)
    unmatched_labels = set(label_names_stripped) - set(image_names_stripped)

    return unmatched_images, unmatched_labels

# Example usage
image_path = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_pont_data\\images\\"
label_path = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_pont_data\\labels\\"

unmatched_images, unmatched_labels = find_unmatched_pairs(image_path, label_path)

print("Images without matching labels:", unmatched_images)
print("Labels without matching images:", unmatched_labels)
'''

r'''
# Move donwloaded images
path = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_okee_2021_data"
labels_path = os.path.join(path, 'labels')
images_path = os.path.join(path, 'images')

# List all files in the source directory
files = os.listdir(path)

# Iterate over each file in the directory
for name in files:
    source_file = os.path.join(path, name)
    # Ensure we only process files, not directories
    if os.path.isfile(source_file):
        if "CIcyano" in name:
            # Move file to the labels folder
            destination = os.path.join(labels_path, name)
        else:
            # Move file to the images folder
            destination = os.path.join(images_path, name)
        shutil.move(source_file, destination)
        print(f"Moved {name} to {destination}")

print("Files moved successfully!")

# Process images in prep.py
'''
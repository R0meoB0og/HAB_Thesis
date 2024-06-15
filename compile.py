# Romain Huet -- Script to compile images from diff lakes into one mega folder for new mega model

import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import tensorflow as tf
import splitfolders as sf

r'''
# folder paths
gb_images = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\green_bay_data\\images\\"
gb_labels = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\green_bay_data\\labels\\"
lo_images = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_okee_2021_data\\images\\"
lo_labels = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_okee_2021_data\\labels\\"
lp_images = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_pont_data\\images\\"
lp_labels = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\lake_pont_data\\labels\\"

# Load files
gb_im_files = os.listdir(gb_images) # load files
gb_lab_files = os.listdir(gb_labels)
lo_im_files = os.listdir(lo_images)
lo_lab_files = os.listdir(lo_labels)
lp_im_files = os.listdir(lp_images)
lp_lab_files = os.listdir(lp_labels)

# remove last 80 days and first 80 days from lake okee (winter, no blooms anyways) to balance dataset
lo_im_files = lo_im_files[80:286]
lo_lab_files = lo_lab_files[80:286]

# Compile files to noaa_data
image_files = gb_im_files + lo_im_files + lp_im_files
label_files = gb_lab_files + lo_lab_files + lp_lab_files
noaa_images = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_test_data\\images\\"
noaa_labels = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_test_data\\labels\\"

# Copy images
for image in image_files:
    # Determine source file path
    if image in gb_im_files:
        source_file = os.path.join(gb_images, image)
    elif image in lo_im_files:
        source_file = os.path.join(lo_images, image)
    elif image in lp_im_files:
        source_file = os.path.join(lp_images, image)

    # Copy the file to the new directory
    if os.path.isfile(source_file):
        destination = os.path.join(noaa_images, image)
        shutil.copy(source_file, destination)

# Copy labels
for label in label_files:
    # Determine source file path
    if label in gb_lab_files:
        source_file = os.path.join(gb_labels, label)
    elif label in lo_lab_files:
        source_file = os.path.join(lo_labels, label)
    elif label in lp_lab_files:
        source_file = os.path.join(lp_labels, label)

    # Copy the file to the new directory
    if os.path.isfile(source_file):
        destination = os.path.join(noaa_labels, label)
        shutil.copy(source_file, destination)
'''

### NOW PREPARE DATA ###

# folder paths
image_string = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_test_data\\images\\"
label_string = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_test_data\\labels\\"

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

r'''
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

all_pairs = filtered_pairs

# Save images and labels
image_dir = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_test\\images\\"
label_dir = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_test\\labels\\"

for i, (image, label) in enumerate(all_pairs):
    image_name = f"image_{i}.png"
    label_name = f"label_{i}.png"
    Image.fromarray(image).save(os.path.join(image_dir, image_name))
    label = (label).astype(np.uint8)  # convert back to 0-255 range
    Image.fromarray(label, mode='L').save(os.path.join(label_dir, label_name))


r'''
# Split test and training datasets
data_folder = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa"
output_folder = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_split"
sf.ratio(data_folder, output=output_folder, seed=1337, ratio=(.75, .25))
'''
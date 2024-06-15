# Romain Huet -- Script to model the U-net.

# First load in common libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import random

# Import image treatment libraries
import imageio
from PIL import Image
import cv2

# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib import colors

# import paths
path_train_im = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_split\\train\\images"
path_train_lab = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_split\\train\\labels"
path_test_im = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_split\\val\\images"
path_test_lab = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\noaa_split\\val\\labels"

# Read the imaging dataset and get items, pairs of images and label maps, as training batches.
def normalise_intensity(image, thres_roi=1.0):
    band_image = np.array(image)
    band_max = np.max(band_image)
    image2 = band_image / band_max
    return image2

class LakeImageSet(Dataset):
    """ Lake image set """
    def __init__(self, image_path, label_path, deploy=False):
        self.image_path = image_path
        self.deploy = deploy
        self.images = []
        self.labels = []

        # Ensure images and labels are sorted and matched
        image_names = sorted(os.listdir(image_path))
        label_names = sorted(os.listdir(label_path))

        assert len(image_names) == len(label_names), "Number of images and labels must be the same."

        for img_name, lbl_name in zip(image_names, label_names):
            # Read the image
            img_path = os.path.join(image_path, img_name)
            lbl_path = os.path.join(label_path, lbl_name)

            image = imageio.imread(img_path)
            label = imageio.imread(lbl_path)

            if label.ndim == 3:
                label = label[:, :, 0]  # Convert to grayscale if necessary

            # Ensure label has a single channel and reshape
            label = np.expand_dims(label, axis=0)

            self.images.append(image)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get an image and perform intensity normalization
        image = normalise_intensity(self.images[idx])

        # Get its label map
        label = self.labels[idx]
        return image, label

    def get_random_batch(self, batch_size):
        # Get a batch of paired images and label maps
        images, labels = [], []

        indices = np.random.randint(0, self.__len__(), batch_size)
        for idx in indices:
            image, label = self.__getitem__(idx)  # normalization

            images.append(image)
            if not self.deploy:
                labels.append(label)

        images = np.stack(images, axis=0)
        labels = np.stack(labels, axis=0) if labels else None

        return images, labels

""" U-net """
class UNet(nn.Module):
    def __init__(self, input_channel=3, output_channel=1, num_filter=16):
        super(UNet, self).__init__()

        # BatchNorm: by default during training this layer keeps running estimates
        # of its computed mean and variance, which are then used for normalization
        # during evaluation.


        # Encoder path
        n = num_filter  # 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU()
        )

        n *= 2  # 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(int(n / 2), n, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU()
        )

        n *= 2  # 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(int(n / 2), n, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU()
        )

        n *= 2  # 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(int(n / 2), n, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU()
        )

        # Decoder path
        ### Insert your code ###
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(n, n//2, stride=2, kernel_size=3, padding=1)
        )

        n = n // 2 # 64
        self.upconv2 = nn.Sequential(
            nn.Conv2d(n*2, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.ConvTranspose2d(n, n//2, stride=2, kernel_size=2)
        )

        n = n // 2 # 32
        self.upconv3 = nn.Sequential(
            nn.Conv2d(n*2, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.ConvTranspose2d(n, n//2, stride=2, kernel_size=2)
        )

        n = n // 2 # 16
        self.upconv4 = nn.Sequential(
            nn.Conv2d(n*2, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, n, kernel_size=3, padding=1),
            nn.BatchNorm2d(n),
            nn.ReLU(),
            nn.Conv2d(n, output_channel, kernel_size=1, padding=0)
        )

        ### End of your code ###

    def forward(self, x):
        # Use the convolutional operators defined above to build the U-net
        # The encoder part is already done for you.
        # You need to complete the decoder part.
        # Encoder
        x = self.conv1(x)
        conv1_skip = x

        x = self.conv2(x)
        conv2_skip = x

        x = self.conv3(x)
        conv3_skip = x

        x = self.conv4(x)

        # Decoder
        ### Insert your code ###
        x = self.upconv1(x)
        x = torch.cat((x, conv3_skip), dim=1)

        x = self.upconv2(x)
        x = torch.cat((x, conv2_skip), dim=1)

        x = self.upconv3(x)
        x = torch.cat((x, conv1_skip), dim=1)

        x = self.upconv4(x)
        ### End of your code ###
        return x

# CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {0}'.format(device))

# Build the model
num_class = 1
model = UNet(input_channel=3, output_channel=num_class, num_filter=16)
model = model.to(device)
params = list(model.parameters())

model_dir = r"C:\\Users\\rdphu\\OneDrive\\Bureau\\RSM\\Thesis\\saved_models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Optimizer
optimizer = optim.Adam(params, lr=1e-3)

# Segmentation loss
criterion = torch.nn.BCEWithLogitsLoss()

### DATASETS ###
train_set = LakeImageSet(path_train_im, path_train_lab)
test_set = LakeImageSet(path_test_im, path_test_lab)

# Train the model
num_iter = 10000
train_batch_size = 16
eval_batch_size = 16
start = time.time()
for it in range(1, 1 + num_iter):
    # Set the modules in training mode, which will have effects on certain modules, e.g. dropout or batchnorm.
    start_iter = time.time()
    model.train()
    print(it)

    # Get a batch of images and labels
    images, labels = train_set.get_random_batch(train_batch_size)
    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
    # Permute the images to match the required input format
    images = images.permute(0, 3, 1, 2)
    logits = model(images)

    # Perform optimisation and print out the training loss
    running_loss = 0
    optimizer.zero_grad()
    loss = criterion(logits, labels.float())
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    # Print running loss
    print(f'Iteration {it}, loss: {loss.item()}')

    # Evaluate
    if it % 100 == 0:
        model.eval()
        # Disabling gradient calculation during reference to reduce memory consumption
        with torch.no_grad():
            # Evaluate on a batch of test images and print out the test loss
            images, labels = test_set.get_random_batch(eval_batch_size)
            images, labels = torch.from_numpy(images), torch.from_numpy(
                labels)
            images = images.permute(0, 3, 1, 2)
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)

            # Forward pass to get model output
            logits = model(images)

            # Calculate the loss using the appropriate data types
            tmp_loss = criterion(logits, labels.float())
            print(f'epoch: {it}, loss: {tmp_loss:.3f}')

    # Save the model
    if it % 5000 == 0:
        torch.save(model.state_dict(), os.path.join(model_dir, 'model_compiled_{0}.pt'.format(it)))
print('Training took {:.3f}s in total.'.format(time.time() - start))


### RESULTS ###
def pixel_accuracy(preds, labels):
    preds = (preds > 0.5).long()  # Convert predictions to binary
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total

def iou(preds, labels, num_classes):
    preds = preds.argmax(dim=1).view(-1)
    labels = labels.view(-1)

    iou_list = []
    present_iou_list = []

    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls

        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection

        if target_inds.long().sum().item() > 0:
            present_iou_list.append(intersection / max(union, 1))

        iou_list.append(intersection / max(union, 1))

    present_iou = np.mean(present_iou_list)
    mean_iou = np.mean(iou_list)

    return mean_iou, present_iou

def dice_coefficient(preds, labels, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    labels = labels.float()

    intersection = (preds * labels).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + labels.sum(dim=(1, 2, 3))

    dice = (2 * intersection) / union
    dice[union == 0] = 1.0  # Avoid division by zero, set dice to 1 if both prediction and label are empty

    return dice.mean().item()

def visualize_results(images, labels, logits, threshold=0.5):
    pred_probs = torch.sigmoid(logits).cpu().detach().numpy()
    images_np = images.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()

    batch_size = pred_probs.shape[0]
    fig, axs = plt.subplots(batch_size, 3, figsize=(15, 5 * batch_size))

    for i in range(batch_size):
        axs[i, 0].imshow(images_np[i].transpose(1, 2, 0))
        axs[i, 0].set_title('Test Image')
        axs[i, 1].imshow(labels_np[i, 0], cmap='gray')
        axs[i, 1].set_title('Ground Truth Segmentation')
        pred_classes = (pred_probs[i, 0] > threshold).astype(np.uint8)
        axs[i, 2].imshow(pred_classes, cmap='gray')
        axs[i, 2].set_title('Predicted Segmentation')

        for ax in axs[i]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def evaluate_model(model, test_set, num_classes, batch_size=4):
    model.eval()
    pixel_accs = []
    ious = []
    dices = []

    with torch.no_grad():
        for _ in range(len(test_set) // batch_size):
            images, labels = test_set.get_random_batch(batch_size)
            images, labels = torch.from_numpy(images), torch.from_numpy(labels)

            if images.ndim == 4 and images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)

            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.long)
            logits = model(images)

            pixel_accs.append(pixel_accuracy(logits, labels))
            mean_iou, present_iou = iou(logits, labels, num_classes)
            ious.append(mean_iou)
            dices.append(dice_coefficient(logits, labels))

            if _ == (len(test_set) // batch_size) - 1:
                visualize_results(images, labels, logits)

    pixel_acc = np.mean(pixel_accs)
    mean_iou = np.mean(ious)
    dice = np.mean(dices)

    return pixel_acc, mean_iou, dice

# Assuming your test dataset is set up
num_classes = 1  # Change according to your dataset
pixel_acc, mean_iou, dice = evaluate_model(model, test_set, num_classes)

print(f'Pixel Accuracy: {pixel_acc:.4f}')
print(f'Mean IoU: {mean_iou:.4f}')
print(f'Dice Coefficient: {dice:.4f}')


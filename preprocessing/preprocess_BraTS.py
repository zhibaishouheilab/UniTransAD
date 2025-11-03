
import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
import random
import glob
import os
from PIL import Image
import imageio

def normalize(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):
    """
    Normalizes the intensity of an image by clipping percentiles and scaling to [0, 1].
    """
    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res = res / res.max()  # Scale to 0-1 range

    return res

def visualize(t1_data,t2_data,flair_data,t1ce_data,gt_data):
    """
    Displays a single slice from each of the four MRI modalities and the ground truth.
    """
    plt.figure(figsize=(8, 8))
    plt.subplot(231)
    plt.imshow(t1_data[:, :], cmap='gray')
    plt.title('Image t1')
    plt.subplot(232)
    plt.imshow(t2_data[:, :], cmap='gray')
    plt.title('Image t2')
    plt.subplot(233)
    plt.imshow(flair_data[:, :], cmap='gray')
    plt.title('Image flair')
    plt.subplot(234)
    plt.imshow(t1ce_data[:, :], cmap='gray')
    plt.title('Image t1ce')
    plt.subplot(235)
    plt.imshow(gt_data[:, :])
    plt.title('GT')
    plt.show()

def visualize_to_gif(t1_data, t2_data, t1ce_data, flair_data):
    """
    Generates GIFs of the brain scans from three different anatomical planes.
    """
    transversal = []
    coronal = []
    sagittal = []
    slice_num = t1_data.shape[2]
    for i in range(slice_num):
        sagittal_plane = np.concatenate((t1_data[:, :, i], t2_data[:, :, i],
                              t1ce_data[:, :, i],flair_data[:, :, i]),axis=1)
        coronal_plane = np.concatenate((t1_data[i, :, :], t2_data[i, :, :],
                              t1ce_data[i, :, :],flair_data[i, :, :]),axis=1)
        transversal_plane = np.concatenate((t1_data[:, i, :], t2_data[:, i, :],
                              t1ce_data[:, i, :],flair_data[:, i, :]),axis=1)
        transversal.append(transversal_plane)
        coronal.append(coronal_plane)
        sagittal.append(sagittal_plane)
    imageio.mimsave("./transversal_plane.gif", transversal, duration=0.01)
    imageio.mimsave("./coronal_plane.gif", coronal, duration=0.01)
    imageio.mimsave("./sagittal_plane.gif", sagittal, duration=0.01)
    return

if __name__ == '__main__':

    # Define paths to the BraTS 2021 dataset
    t1_list = sorted(glob.glob(
        '/mnt/sdb/zq/brain_sas_baseline/datasets/BraTS2021_TrainingData/*/*t1.*'))
    t2_list = sorted(glob.glob(
        '/mnt/sdb/zq/brain_sas_baseline/datasets/BraTS2021_TrainingData/*/*t2.*'))
    t1ce_list = sorted(glob.glob(
        '/mnt/sdb/zq/brain_sas_baseline/datasets/BraTS2021_TrainingData/*/*t1ce.*'))
    flair_list = sorted(glob.glob(
        '/mnt/sdb/zq/brain_sas_baseline/datasets/BraTS2021_TrainingData/*/*flair.*'))
    gt_list = sorted(glob.glob(
        '/mnt/sdb/zq/brain_sas_baseline/datasets/BraTS2021_TrainingData/*/*seg.*'))

    # Calculate split points for a 7:2:1 (train:test:eval) ratio
    data_len = len(gt_list)
    train_split_idx = int(data_len * 0.7)
    eval_split_idx = train_split_idx + int(data_len * 0.1)
    # The remaining ~20% will be for testing

    # Define output directories
    train_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train/'
    test_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/test/'
    eval_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/eval/'

    # Create directories if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(eval_path, exist_ok=True)

    for i, (t1_path, t2_path, t1ce_path, flair_path, gt_path) in enumerate(zip(t1_list, t2_list, t1ce_list, flair_list, gt_list)):

        print('Preprocessing the', i + 1, 'th subject')

        # Load NIfTI images
        t1_img = nib.load(t1_path)
        t2_img = nib.load(t2_path)
        flair_img = nib.load(flair_path)
        t1ce_img = nib.load(t1ce_path)
        gt_img = nib.load(gt_path)

        # Convert to numpy arrays
        t1_data = t1_img.get_fdata()
        t2_data = t2_img.get_fdata()
        flair_data = flair_img.get_fdata()
        t1ce_data = t1ce_img.get_fdata()
        gt_data = gt_img.get_fdata()
        gt_data = gt_data.astype(np.uint8)
        gt_data[gt_data == 4] = 3  # Remap label 4 to 3

        # Normalize each modality
        t1_data = normalize(t1_data)
        t2_data = normalize(t2_data)
        t1ce_data = normalize(t1ce_data)
        flair_data = normalize(flair_data)

        # Stack modalities and ground truth into a single tensor
        tensor = np.stack([t1_data, t2_data, t1ce_data, flair_data, gt_data])
        # Rotate for correct anatomical orientation
        tensor = np.rot90(tensor, k=2, axes=(1, 2))

        # Split and save data into respective folders
        if i < train_split_idx:
            # Save to training set
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                np.save(train_path + str(60 * i + j + 1) + '.npy', Tensor)
        elif i < eval_split_idx:
            # Save to evaluation set
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                # Offset index to start file numbering from 1
                np.save(eval_path + str(60 * (i - train_split_idx) + j + 1) + '.npy', Tensor)
        else:
            # Save to test set
            for j in range(60):
                Tensor = tensor[:, 10:210, 25:225, 50 + j]
                # Offset index to start file numbering from 1
                np.save(test_path + str(60 * (i - eval_split_idx) + j + 1) + '.npy', Tensor)
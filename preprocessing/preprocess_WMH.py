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
    Normalizes a NIfTI image array.
    """
    if mask is None:
        mask = image != image[0, 0, 0]
    
    # Ensure mask is not all False, which can happen with empty resized slices
    if not np.any(mask):
        return image # Return the original image if mask is empty

    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    
    # Avoid division by zero if the max is 0
    max_val = res.max()
    if max_val > 0:
        res = res / max_val  # 0-1
    
    return res


if __name__ == '__main__':
    base_folder = '/mnt/sdb/zq/brain_sas_baseline/datasets/WMH_Registered'
    
    # Get all image paths
    flair_list = sorted(glob.glob(os.path.join(base_folder, 'FLAIR/*.nii.gz')))
    t1_list = sorted(glob.glob(os.path.join(base_folder, 'T1/*.nii.gz')))
    label_list = sorted(glob.glob(os.path.join(base_folder, 'label/*.nii.gz')))
    
    train_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/train/'
    test_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/test/'

    # Create dictionaries with a common prefix as key
    t1_dict = {os.path.basename(path).replace('.nii.gz', ''): path for path in t1_list}
    flair_dict = {os.path.basename(path).replace('.nii.gz', ''): path for path in flair_list}
    label_dict = {os.path.basename(path).replace('.nii.gz', ''): path for path in label_list}
    
    # Find common prefixes among the three modalities
    common_prefixes = sorted(list(set(t1_dict.keys()) & set(flair_dict.keys()) & set(label_dict.keys())))
    
    print(f"Found {len(common_prefixes)} subjects with all three modalities.")
    
    # Create save directories
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # Split into training and testing sets (ratio can be adjusted)
    train_ratio = 0.9 # Using all for training as in the original code
    train_len = int(len(common_prefixes) * train_ratio)

    for i, prefix in enumerate(common_prefixes):
        print(f'Preprocessing subject {i+1}/{len(common_prefixes)}: {prefix}')
        
        # Get paths for the three modalities
        t1_path = t1_dict[prefix]
        flair_path = flair_dict[prefix]
        label_path = label_dict[prefix]
        
        # Load NIfTI images
        t1_img = nib.load(t1_path)
        flair_img = nib.load(flair_path)
        label_img = nib.load(label_path)
        
        # Get image data arrays
        label_data = label_img.get_fdata().astype(np.uint8)
        flair_data = flair_img.get_fdata()
        t1_data = t1_img.get_fdata()
        
        # Binarize the label to keep only white matter hyperintensities (label=1)
        label_data[label_data != 1] = 0
        
        # Normalize the RESIZED images
        flair_data_normalized = normalize(flair_data)
        t1_data_normalized = normalize(t1_data)
        
        # Stack the processed arrays into a single tensor (modalities, height, width, depth)
        # Note: The data type of the label is changed to match the others for stacking
        tensor = np.stack([t1_data_normalized, flair_data_normalized, label_data.astype(float)])
        print(f"Shape after resizing and stacking: {tensor.shape}")

        # 保存切片
        if i < train_len:
            save_path = train_path
            index_offset = i * 60
        else:
            save_path = test_path
            index_offset = (i - train_len) * 60
            
        for j in range(60):
            # 提取切片
            tensor_slice = tensor[:, 10:210, 25:225, 50 + j]
            # 保存为npy文件
            np.save(f"{save_path}{index_offset + j + 1}.npy", tensor_slice)
import numpy as np
import nibabel as nib
import glob
import os
from tqdm import tqdm

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


if __name__ == '__main__':
    # --- 1. DEFINE PATHS ---
    # Base folder should be the output of the previous registration script
    base_folder = '/mnt/sdb/zq/brain_sas_baseline/datasets/ISLES-2022_nii_registered'
    
    # Output folders for the .npy slices
    train_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/train/'
    test_path = '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/test/'

    # --- 2. FIND AND GROUP IMAGE PATHS ---
    adc_list = sorted(glob.glob(os.path.join(base_folder, 'ADC/*.nii.gz')))
    dwi_list = sorted(glob.glob(os.path.join(base_folder, 'DWI/*.nii.gz')))
    flair_list = sorted(glob.glob(os.path.join(base_folder, 'FLAIR/*.nii.gz')))
    # The label folder contains the mask files (*_msk.nii.gz)
    msk_list = sorted(glob.glob(os.path.join(base_folder, 'label/*.nii.gz')))
    
    # Create dictionaries with a common prefix as key for robust matching
    adc_dict = {os.path.basename(path).replace('_adc.nii.gz', ''): path for path in adc_list}
    dwi_dict = {os.path.basename(path).replace('_dwi.nii.gz', ''): path for path in dwi_list}
    flair_dict = {os.path.basename(path).replace('_flair.nii.gz', ''): path for path in flair_list}
    msk_dict = {os.path.basename(path).replace('_msk.nii.gz', ''): path for path in msk_list}
    
    # Find common prefixes among the four modalities to ensure we only process complete sets
    common_prefixes = sorted(list(
        set(adc_dict.keys()) & set(dwi_dict.keys()) & set(flair_dict.keys()) & set(msk_dict.keys())
    ))
    
    print(f"Found {len(common_prefixes)} subjects with all four required modalities (ADC, DWI, FLAIR, MSK).")
    
    # Create save directories if they don't exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    
    # --- 3. SPLIT DATA AND PROCESS ---
    train_ratio = 0.9
    train_len = int(len(common_prefixes) * train_ratio)

    # Process each subject
    for i, prefix in enumerate(tqdm(common_prefixes, desc="Processing Subjects")):
        
        # Get paths for the four modalities
        adc_path = adc_dict[prefix]
        dwi_path = dwi_dict[prefix]
        flair_path = flair_dict[prefix]
        msk_path = msk_dict[prefix]
        
        # Load NIfTI images
        adc_img = nib.load(adc_path)
        dwi_img = nib.load(dwi_path)
        flair_img = nib.load(flair_path)
        msk_img = nib.load(msk_path)
        
        # Get image data arrays
        adc_data = adc_img.get_fdata()
        dwi_data = dwi_img.get_fdata()
        flair_data = flair_img.get_fdata()
        # The mask is integer-based (0 or 1)
        msk_data = msk_img.get_fdata().astype(np.uint8)
        
        # Normalize the three intensity modalities
        adc_norm = normalize(adc_data)
        dwi_norm = normalize(dwi_data)
        flair_norm = normalize(flair_data)
        
        # Stack the processed arrays into a single 4-channel tensor
        # (modalities, height, width, depth)
        tensor = np.stack([
            adc_norm, 
            dwi_norm, 
            flair_norm, 
            msk_data.astype(float) # Cast mask to float for stacking
        ])
        
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
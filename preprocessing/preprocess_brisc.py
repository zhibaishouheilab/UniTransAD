import os
import glob
import argparse
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from .utils import normalize_intensity

def process_brisc(args):
    img_dir = os.path.join(args.src_path, 'images_nii')
    bmask_dir = os.path.join(args.src_path, 'masks_nii') # Brain mask
    lmask_dir = os.path.join(args.src_path, 'labels_nii') # Lesion mask
    
    img_list = sorted(glob.glob(os.path.join(img_dir, '*.nii.gz')))
    img_dict = {os.path.basename(p).replace('_0000.nii.gz', ''): p for p in img_list}
    
    # Find commons logic ... (same as original)
    bmask_list = sorted(glob.glob(os.path.join(bmask_dir, '*.nii.gz')))
    bmask_dict = {os.path.basename(p).replace('.nii.gz', ''): p for p in bmask_list}
    
    lmask_list = sorted(glob.glob(os.path.join(lmask_dir, '*.nii.gz')))
    lmask_dict = {os.path.basename(p).replace('.nii.gz', ''): p for p in lmask_list}
    
    common_ids = sorted(list(set(img_dict) & set(bmask_dict) & set(lmask_dict)))
    print(f"Found {len(common_ids)} subjects in BRISC.")

    random.seed(args.seed)
    random.shuffle(common_ids)

    train_len = int(len(common_ids) * 0.9)
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    train_counter, test_counter = 1, 1

    for i, sub_id in enumerate(tqdm(common_ids, desc="Processing BRISC")):
        # Load 2D data (squeeze extra dims)
        img = np.squeeze(nib.load(img_dict[sub_id]).get_fdata())
        bmask = np.squeeze(nib.load(bmask_dict[sub_id]).get_fdata()).astype(np.uint8)
        lmask = np.squeeze(nib.load(lmask_dict[sub_id]).get_fdata()).astype(np.uint8)
        
        # Rotate 90
        img = np.rot90(img, k=1)
        bmask = np.rot90(bmask, k=1)
        lmask = np.rot90(lmask, k=1)
        
        # Skull strip & Normalize
        img_stripped = img * bmask
        img_norm = normalize_intensity(img_stripped, mask=bmask)
        
        # Stack (2, H, W) -> [T1, Label]
        tensor = np.stack([img_norm, lmask.astype(float)])
        
        if i < train_len:
            np.save(os.path.join(args.train_output, f"{train_counter}.npy"), tensor)
            train_counter += 1
        else:
            np.save(os.path.join(args.test_output, f"{test_counter}.npy"), tensor)
            test_counter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="Path to BRISC segmentation_task folder")
    parser.add_argument("--train_output", type=str, default="./dataset/npy/BRISC/train")
    parser.add_argument("--test_output", type=str, default="./dataset/npy/BRISC/test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    process_brisc(args)
import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from .utils import normalize_intensity, save_slices

def process_wmh(args):
    # Paths assuming 'WMH_Registered' structure with T1, FLAIR, label subfolders
    t1_dict = {os.path.basename(p).replace('.nii.gz', ''): p for p in sorted(glob.glob(os.path.join(args.src_path, 'T1/*.nii.gz')))}
    flair_dict = {os.path.basename(p).replace('.nii.gz', ''): p for p in sorted(glob.glob(os.path.join(args.src_path, 'FLAIR/*.nii.gz')))}
    label_dict = {os.path.basename(p).replace('.nii.gz', ''): p for p in sorted(glob.glob(os.path.join(args.src_path, 'label/*.nii.gz')))}
    
    common = sorted(list(set(t1_dict) & set(flair_dict) & set(label_dict)))
    print(f"Found {len(common)} subjects in WMH.")
    
    train_len = int(len(common) * 0.9)
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    for i, prefix in enumerate(tqdm(common, desc="Processing WMH")):
        t1 = normalize_intensity(nib.load(t1_dict[prefix]).get_fdata())
        flair = normalize_intensity(nib.load(flair_dict[prefix]).get_fdata())
        lbl = nib.load(label_dict[prefix]).get_fdata().astype(np.uint8)
        
        # Keep only label 1 (WMH)
        lbl[lbl != 1] = 0
        
        # Stack: (3, H, W, D) -> [T1, FLAIR, Label]
        tensor = np.stack([t1, flair, lbl.astype(float)])
        
        if i < train_len:
            save_slices(tensor, args.train_output, i * 60)
        else:
            save_slices(tensor, args.test_output, (i - train_len) * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="Path to WMH_Registered")
    parser.add_argument("--train_output", type=str, default="./dataset/npy/WMH/train")
    parser.add_argument("--test_output", type=str, default="./dataset/npy/WMH/test")
    args = parser.parse_args()
    process_wmh(args)
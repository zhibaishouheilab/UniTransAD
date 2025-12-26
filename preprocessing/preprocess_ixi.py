import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from .utils import normalize_intensity, save_slices

def process_ixi(args):
    # Using 'Registered' folder structure
    t1_dict = {os.path.basename(p).replace('-T1_registered.nii.gz', ''): p for p in sorted(glob.glob(os.path.join(args.src_path, 'T1/*.nii.gz')))}
    t2_dict = {os.path.basename(p).replace('-T2_registered.nii.gz', ''): p for p in sorted(glob.glob(os.path.join(args.src_path, 'T2/*.nii.gz')))}
    pd_dict = {os.path.basename(p).replace('-PD_registered.nii.gz', ''): p for p in sorted(glob.glob(os.path.join(args.src_path, 'PD/*.nii.gz')))}
    
    common = sorted(list(set(t1_dict) & set(t2_dict) & set(pd_dict)))
    print(f"Found {len(common)} subjects in IXI.")
    
    # 100% for training (Healthy dataset)
    os.makedirs(args.train_output, exist_ok=True)

    for i, prefix in enumerate(tqdm(common, desc="Processing IXI")):
        t1 = normalize_intensity(nib.load(t1_dict[prefix]).get_fdata())
        t2 = normalize_intensity(nib.load(t2_dict[prefix]).get_fdata())
        pd = normalize_intensity(nib.load(pd_dict[prefix]).get_fdata())
        
        # Stack: (3, H, W, D) -> [T1, T2, PD]
        tensor = np.stack([t1, t2, pd])
        
        save_slices(tensor, args.train_output, i * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="Path to IXI/Registered")
    parser.add_argument("--train_output", type=str, default="./dataset/npy/IXI/train")
    args = parser.parse_args()
    process_ixi(args)
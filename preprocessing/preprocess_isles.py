import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from .utils import normalize_intensity, save_slices

def process_isles(args):
    # Paths assuming standardized registration output structure
    adc_list = sorted(glob.glob(os.path.join(args.src_path, 'ADC/*.nii.gz')))
    dwi_list = sorted(glob.glob(os.path.join(args.src_path, 'DWI/*.nii.gz')))
    flair_list = sorted(glob.glob(os.path.join(args.src_path, 'FLAIR/*.nii.gz')))
    msk_list = sorted(glob.glob(os.path.join(args.src_path, 'label/*.nii.gz')))
    
    # Clean logic to match prefixes
    def get_prefix(path, suffix):
        return os.path.basename(path).replace(suffix, '')

    adc_dict = {get_prefix(p, '_adc.nii.gz'): p for p in adc_list}
    dwi_dict = {get_prefix(p, '_dwi.nii.gz'): p for p in dwi_list}
    flair_dict = {get_prefix(p, '_flair.nii.gz'): p for p in flair_list}
    msk_dict = {get_prefix(p, '_msk.nii.gz'): p for p in msk_list}
    
    common = sorted(list(set(adc_dict) & set(dwi_dict) & set(flair_dict) & set(msk_dict)))
    print(f"Found {len(common)} complete subjects in ISLES.")
    
    train_len = int(len(common) * 0.9)
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    for i, prefix in enumerate(tqdm(common, desc="Processing ISLES")):
        adc = normalize_intensity(nib.load(adc_dict[prefix]).get_fdata())
        dwi = normalize_intensity(nib.load(dwi_dict[prefix]).get_fdata())
        flair = normalize_intensity(nib.load(flair_dict[prefix]).get_fdata())
        msk = nib.load(msk_dict[prefix]).get_fdata().astype(np.uint8)
        
        # Stack: (4, H, W, D) -> [ADC, DWI, FLAIR, Label]
        tensor = np.stack([adc, dwi, flair, msk.astype(float)])
        
        if i < train_len:
            save_slices(tensor, args.train_output, i * 60)
        else:
            save_slices(tensor, args.test_output, (i - train_len) * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="Path to Registered ISLES data")
    parser.add_argument("--train_output", type=str, default="./dataset/npy/ISLES/train")
    parser.add_argument("--test_output", type=str, default="./dataset/npy/ISLES/test")
    args = parser.parse_args()
    process_isles(args)
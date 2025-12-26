import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from .utils import normalize_intensity, save_slices

def process_msd(args):
    images_dir = os.path.join(args.src_path, 'imagesTr')
    labels_dir = os.path.join(args.src_path, 'labelsTr')
    
    image_list = sorted(glob.glob(os.path.join(images_dir, '*.nii.gz')))
    label_list = sorted(glob.glob(os.path.join(labels_dir, '*.nii.gz')))
    
    # Match files
    img_dict = {os.path.basename(p): p for p in image_list}
    lbl_dict = {os.path.basename(p): p for p in label_list}
    common_files = sorted(list(set(img_dict.keys()) & set(lbl_dict.keys())))
    
    print(f"Found {len(common_files)} subjects in MSD.")
    
    train_len = int(len(common_files) * 0.9)
    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)

    for i, filename in enumerate(tqdm(common_files, desc="Processing MSD")):
        img_data = nib.load(img_dict[filename]).get_fdata() # (H, W, D, C) -> (240, 240, 155, 4)
        lbl_data = nib.load(lbl_dict[filename]).get_fdata().astype(np.uint8) # (H, W, D)

        # MSD Channel Order: [FLAIR, T1w, T1gd, T2w] -> Transpose to (C, H, W, D)
        img_data = img_data.transpose(3, 0, 1, 2)
        
        normalized_channels = [normalize_intensity(img_data[c]) for c in range(4)]
        
        # Expand Label to (1, H, W, D)
        lbl_expanded = np.expand_dims(lbl_data, axis=0)
        
        # Stack: (5, H, W, D)
        tensor = np.concatenate([np.array(normalized_channels), lbl_expanded.astype(float)], axis=0)

        if i < train_len:
            save_slices(tensor, args.train_output, i * 60)
        else:
            save_slices(tensor, args.test_output, (i - train_len) * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="Path to MSD_BrainTumour (containing imagesTr)")
    parser.add_argument("--train_output", type=str, default="./dataset/npy/MSD/train")
    parser.add_argument("--test_output", type=str, default="./dataset/npy/MSD/test")
    args = parser.parse_args()
    process_msd(args)
import os
import glob
import shutil
import argparse
import numpy as np
from tqdm import tqdm

def select_normal(args):
    os.makedirs(args.dest_path, exist_ok=True)
    npy_files = glob.glob(os.path.join(args.source_path, "*.npy"))
    
    print(f"Scanning {len(npy_files)} files for normal slices (empty labels)...")
    
    selected_count = 0
    for f in tqdm(npy_files):
        try:
            data = np.load(f)
            label = data[-1]
            
            # 逻辑修复：如果 label 中没有任何像素 > 0，则是正常切片
            if not np.any(label > 0):
                shutil.copy(f, args.dest_path)
                selected_count += 1
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    print(f"Selected {selected_count} normal slices out of {len(npy_files)}.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", type=str, required=True, help="Path to npy files (e.g. BraTS2021/train)")
    parser.add_argument("--dest_path", type=str, required=True, help="Path to save normal slices (e.g. BraTS2021/train_normal)")
    args = parser.parse_args()
    select_normal(args)
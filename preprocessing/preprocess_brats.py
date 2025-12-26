import os
import glob
import argparse
import numpy as np
import nibabel as nib
from tqdm import tqdm
from .utils import normalize_intensity, save_slices

def process_brats(args):
    # 路径通配符
    t1_pattern = os.path.join(args.src_path, '*/*t1.nii.gz')
    t1_list = sorted(glob.glob(t1_pattern))
    
    # 简单校验
    if len(t1_list) == 0:
        # 尝试匹配无后缀的情况 (原始 BraTS 格式)
        t1_pattern = os.path.join(args.src_path, '*/*t1.*')
        t1_list = sorted(glob.glob(t1_pattern))
    
    print(f"Found {len(t1_list)} subjects in {args.src_path}")

    # 划分数据集
    data_len = len(t1_list)
    train_split = int(data_len * 0.7)
    eval_split = train_split + int(data_len * 0.1)

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)
    os.makedirs(args.eval_output, exist_ok=True)

    for i, t1_p in enumerate(tqdm(t1_list, desc="Processing BraTS")):
        # 根据 t1 路径推导其他模态路径
        # 假设结构为 Subject/Subject_t1.nii.gz
        parent_dir = os.path.dirname(t1_p)
        file_prefix = os.path.basename(t1_p).split('t1')[0] # 获取 Subject_
        ext = '.nii.gz' if t1_p.endswith('.nii.gz') else '.nii'

        t2_p = os.path.join(parent_dir, f"{file_prefix}t2{ext}")
        ce_p = os.path.join(parent_dir, f"{file_prefix}t1ce{ext}")
        fl_p = os.path.join(parent_dir, f"{file_prefix}flair{ext}")
        gt_p = os.path.join(parent_dir, f"{file_prefix}seg{ext}")

        if not os.path.exists(gt_p):
            print(f"Skipping {file_prefix}: GT not found.")
            continue

        # Load
        t1 = normalize_intensity(nib.load(t1_p).get_fdata())
        t2 = normalize_intensity(nib.load(t2_p).get_fdata())
        ce = normalize_intensity(nib.load(ce_p).get_fdata())
        fl = normalize_intensity(nib.load(fl_p).get_fdata())
        
        gt = nib.load(gt_p).get_fdata().astype(np.uint8)
        gt[gt == 4] = 3  # Remap label 4 -> 3

        # Stack (5, H, W, D) -> (T1, T2, T1ce, Flair, GT)
        tensor = np.stack([t1, t2, ce, fl, gt])
        
        # BraTS 通常需要旋转以匹配标准视角
        tensor = np.rot90(tensor, k=2, axes=(1, 2))

        # Save
        if i < train_split:
            save_slices(tensor, args.train_output, i * 60)
        elif i < eval_split:
            save_slices(tensor, args.eval_output, (i - train_split) * 60)
        else:
            save_slices(tensor, args.test_output, (i - eval_split) * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, required=True, help="Path to BraTS2021_TrainingData")
    parser.add_argument("--train_output", type=str, default="./dataset/npy/BraTS2021/train")
    parser.add_argument("--test_output", type=str, default="./dataset/npy/BraTS2021/test")
    parser.add_argument("--eval_output", type=str, default="./dataset/npy/BraTS2021/eval")
    args = parser.parse_args()
    process_brats(args)
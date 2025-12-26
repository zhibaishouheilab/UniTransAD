import os
import sys
import re
import json
import argparse
import numpy as np
import torch
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
import cv2
from scipy import ndimage
from multiprocessing import Pool, cpu_count
from argparse import Namespace

# Imports from refactored modules
from data.dataset import get_dataloader
from models.unitransad import UniTransAD

# Reuse logic from tune.py for inference
# Note: This works because we added project_root to sys.path above
from tune import run_inference_for_dataset, parallel_median_filter, apply_connected_component, load_and_stack, cleanup, compute_dice

def compute_auroc(p, t): return roc_auc_score(t.flatten(), p.flatten())
def compute_aupr(p, t):
    precision, recall, _ = precision_recall_curve(t.flatten(), p.flatten())
    return auc(recall, precision)

def compute_dice_5_fold_cv(preds, targs, cc_size, n_thresh=100, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_dices = []
    indices = np.arange(preds.shape[0])
    for train_idx, test_idx in kf.split(indices):
        train_preds, train_targets = preds[train_idx], targs[train_idx]
        test_preds, test_targets = preds[test_idx], targs[test_idx]
        
        tmin, tmax = train_preds.min(), train_preds.max()
        best_thresh, best_dice_train = tmin, -1.0
        
        # Threshold search on train fold
        for t in torch.linspace(tmin, tmax, n_thresh):
            pred_bin = (train_preds > t).float()
            proc_bin = torch.stack([apply_connected_component(b.cpu().numpy(), cc_size) for b in pred_bin]).to(pred_bin.device)
            dice = compute_dice(proc_bin, train_targets)
            if dice > best_dice_train: best_dice_train, best_thresh = dice, t
            
        # Apply to test fold
        test_bin = (test_preds > best_thresh).float()
        proc_test_bin = torch.stack([apply_connected_component(b.cpu().numpy(), cc_size) for b in test_bin]).to(pred_bin.device)
        dice_on_test = compute_dice(proc_test_bin, test_targets)
        fold_dices.append(dice_on_test.item())
        
    return np.mean(fold_dices), np.std(fold_dices)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning_results_json", required=True)
    parser.add_argument("--results_dir", default="./eval_results")
    parser.add_argument("--sample_size", default="all")
    parser.add_argument("--cuda_device", default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.cuda_device)
    os.makedirs(args.results_dir, exist_ok=True)
    
    with open(args.tuning_results_json) as f:
        tuning = json.load(f)
    
    best_model_path = tuning["best_model_path"]
    
    # Config - MUST match training & tuning config
    # Explicitly setting patch_size=8 to avoid shape mismatch
    opt = Namespace(img_size=256, dim_encoder=128, depth=12, num_heads=16, batch_size=8,
                    dim_decoder=64, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.0, 
                    patch_size=8, 
                    ema_momentum=0.1)

    # Dataset Configs (PLACEHOLDERS - Update paths as needed!)
    datasets = {
        'BraTS2021': {'modality': ['t1', 't2', 't1ce', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/test_edge','train_data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal', 'train_edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal_edge', 'fixed_size': 64, 'img_size': 256},
        'MSLUB': {'modality': ['t1', 't2','flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/train_edge', 'fixed_size': 256, 'img_size': 256},
        'ISLES2022': {'modality': ['adc', 'dwi', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/train_edge', 'fixed_size': 128, 'img_size': 256},
        'WMH': {'modality': ['t1', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/train_edge', 'fixed_size': 128, 'img_size': 256},
        'BRISC': {'modality': ['t1'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BRISC/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BRISC/train_edge', 'fixed_size': 64, 'img_size': 256},
        'MSD': {'modality': ['flair','t1','t1ce','t2'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSD_BrainTumour/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSD_BrainTumour/train_edge','fixed_size': 64, 'img_size': 256},
    }
    
    for name, config in datasets.items():
        config['name'] = name
        config['output_path'] = os.path.join(args.results_dir, f"inf_{name}_val")

    # Load DSPM
    # [FIX] Instantiate with correct architecture params (patch_size=8)
    temp = UniTransAD(
        img_size=opt.img_size, 
        patch_size=opt.patch_size,   # <--- [FIX] Critical for loading state_dict
        embed_dim=opt.dim_encoder, 
        depth=opt.depth, 
        num_heads=opt.num_heads,
        decoder_embed_dim=opt.dim_decoder, 
        decoder_depth=opt.decoder_depth, 
        decoder_num_heads=opt.decoder_num_heads,
        mlp_ratio=opt.mlp_ratio,
        ema_momentum=opt.ema_momentum
    ).to(device)

    sd = torch.load(best_model_path, map_location=device)
    temp.load_state_dict(sd, strict=False)
    dspm_feats = {}
    for mod in ['t1', 't1ce', 't2', 'flair']:
        mean_key = f'dspm_{mod}_mean' if f'dspm_{mod}_mean' in sd else f'brats_{mod}_ema_mean'
        std_key = f'dspm_{mod}_std' if f'dspm_{mod}_std' in sd else f'brats_{mod}_ema_std'
        dspm_feats[mod] = {'mean': sd.get(mean_key).to(device), 'std': sd.get(std_key).to(device)}
    del temp

    log_path = os.path.join(args.results_dir, "final_report.txt")
    
    for name, config in datasets.items():
        if name not in tuning["datasets"]: continue
        params = tuning["datasets"][name]
        seq, mf, cc = params["best_sequence"], params["best_median_filter_size"], params["best_connected_component_size"]
        
        print(f"Evaluating {name} using sequence {seq}...")
        run_inference_for_dataset(best_model_path, config, dspm_feats, device, opt, args.sample_size)
        
        # Load Dual-Level Map (mix_err) and Pixel Map (err)
        for mode in ["_mix_err", "_err"]:
            targets, scores = load_and_stack(config['output_path'], config['fixed_size'], re.compile(fr'^\d+{mode}\.npz$'))
            if targets is None: continue
            
            preds = scores[seq]
            preds_np = preds.cpu().numpy().squeeze()
            
            # Filter
            filt_np = parallel_median_filter(preds_np, mf)
            filt_preds = torch.from_numpy(filt_np).unsqueeze(1).to(device)
            
            # Metrics
            dice_mu, dice_std = compute_dice_5_fold_cv(filt_preds, targets, cc)
            auroc = compute_auroc(filt_preds.cpu().numpy(), targets.cpu().numpy())
            auprc = compute_aupr(filt_preds.cpu().numpy(), targets.cpu().numpy())
            
            res_str = f"Dataset: {name} | Mode: {mode} | Seq: {seq}\n  Dice: {dice_mu:.4f} +/- {dice_std:.4f}\n  AUROC: {auroc:.4f}\n  AUPRC: {auprc:.4f}\n"
            print(res_str)
            with open(log_path, "a") as f: f.write(res_str + "\n")
            
        cleanup(config['output_path'])

if __name__ == "__main__":
    main()
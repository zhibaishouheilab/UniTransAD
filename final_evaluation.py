import os
import re
import json
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import KFold
import cv2
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
from scipy import ndimage
from multiprocessing import Pool, cpu_count
import pandas as pd  # [NEW] 增加 pandas 依赖

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils.MM_loader import get_maeloader
from model.MMPC_any import MultiModalPatchMAE

# --- 推理逻辑所需的辅助函数 ---
def content_cosine_similarity(zA, zB, maskA=None, maskB=None):
    B, L, D = zA.shape
    sim_matrix = F.cosine_similarity(zA.unsqueeze(2), zB.unsqueeze(1), dim=-1)
    valid_2d = (maskA.unsqueeze(2) * maskB.unsqueeze(1)) > 0 if maskA is not None and maskB is not None else torch.ones_like(sim_matrix, dtype=torch.bool)
    sim_matrix[~valid_2d] = 0.0
    sim_matrix_norm = torch.zeros_like(sim_matrix)
    eps = 1e-8
    for b in range(B):
        sub_mat, valid_mask = sim_matrix[b], valid_2d[b]
        in_range_vals = sub_mat[valid_mask]
        if in_range_vals.numel() > 0:
            min_val, max_val = in_range_vals.min(), in_range_vals.max()
            denom = (max_val - min_val).clamp_min(eps)
            tmp = sub_mat.clone()
            tmp[valid_mask] = (tmp[valid_mask] - min_val) / denom
            sim_matrix_norm[b] = tmp
    return sim_matrix_norm

def get_diagonal_map(sim_matrix):
    B, L, _ = sim_matrix.shape
    diag = torch.diagonal(sim_matrix, dim1=1, dim2=2)
    side = int(L**0.5)
    diag_map = diag.view(B, side, side)
    diag_map_norm = torch.zeros_like(diag_map)
    eps = 1e-8
    for b in range(B):
        sub = diag_map[b]
        pos_vals = sub[sub > 0]
        if pos_vals.numel() > 0:
            min_val, max_val = pos_vals.min(), pos_vals.max()
            denom = (max_val - min_val).clamp_min(eps)
            tmp = sub.clone()
            tmp[sub > 0] = (sub[sub > 0] - min_val) / denom
            diag_map_norm[b] = tmp
    return diag_map_norm

def save_npz_data(path, **kwargs):
    np.savez(path, **kwargs)

# [新增] 计算 BraTS 平均风格的函数
def compute_train_style_stats(brats_config, checkpoint_path, device, opt):
    print(f"Computing universal style stats from BraTS2021 train set using model: {os.path.basename(checkpoint_path)}...")
    mae = MultiModalPatchMAE(
        img_size=opt.img_size, patch_size=opt.patch_size, embed_dim=opt.dim_encoder,
        depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth,
        decoder_num_heads=opt.decoder_num_heads, mlp_ratio=opt.mlp_ratio,
        norm_layer=nn.LayerNorm
    ).to(device)
    mae.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    mae.eval()
    
    train_loader = get_maeloader(
        batchsize=opt.batch_size, shuffle=False, pin_memory=True, img_size=opt.img_size,
        img_root=brats_config['train_data_root'], edge_root=brats_config['train_edge_root'],
        num_workers=16, augment=False, if_addlabel=True, modality=brats_config['modality']
    )
    
    n_modalities = len(brats_config['modality'])
    all_means = [[] for _ in range(n_modalities)]
    all_stds = [[] for _ in range(n_modalities)]
    
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Computing style stats", leave=False):
            modals = [m.to(device, dtype=torch.float) for m in batch[:n_modalities]]
            for i, modal in enumerate(modals):
                embed = mae.patch_embed(modal)
                embed = torch.cat([mae.cls_token.expand(embed.shape[0], -1, -1), embed], dim=1) + mae.pos_embed
                style_mean, style_std = mae.compute_style_mean_std(embed)
                all_means[i].append(style_mean.cpu())
                all_stds[i].append(style_std.cpu())
    
    style_means = [torch.cat(m).mean(dim=0).to(device) for m in all_means]
    style_stds = [torch.cat(s).mean(dim=0).to(device) for s in all_stds]
    
    brats_modalities_list = ['t1', 't1ce', 't2', 'flair']
    brats_style_features = {
        mod_name: {'mean': mean, 'std': std}
        for mod_name, mean, std in zip(brats_modalities_list, style_means, style_stds)
    }
    print("Universal style features computed.")
    return brats_style_features

# [MODIFIED] 基于循环翻译的推理函数 (高效批处理版本)
def run_inference_for_dataset(model_path, dataset_config, brats_style_features, device, opt):
    print(f"\n--- Running CYCLE inference for dataset: {dataset_config['name']} on validation set ---")
    img_save_path = dataset_config['output_path']
    os.makedirs(img_save_path, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())

    mae = MultiModalPatchMAE(
        img_size=opt.img_size, patch_size=opt.patch_size, embed_dim=opt.dim_encoder,
        depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth,
        decoder_num_heads=opt.decoder_num_heads, mlp_ratio=opt.mlp_ratio,
        norm_layer=nn.LayerNorm
    ).to(device)
    mae.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    mae.eval()

    val_loader = get_maeloader(
        batchsize=opt.batch_size, shuffle=False, pin_memory=True, img_size=dataset_config['img_size'],
        img_root=dataset_config['data_root'], num_workers=16, augment=False, if_addlabel=True,
        modality=dataset_config['modality'], edge_root=dataset_config['edge_root']
    )
    modality_names = {idx: name for idx, name in enumerate(dataset_config['modality'])}
    n_modalities = len(dataset_config['modality'])
    brats_modalities_list = ['t1', 't1ce', 't2', 'flair']

    for i, batch in enumerate(tqdm(val_loader, desc=f"Cycle Inference on {dataset_config['name']}", leave=False)):
        modals = [batch[j].to(device, dtype=torch.float) for j in range(n_modalities)]
        edges = [batch[j + n_modalities].to(device, dtype=torch.float) for j in range(n_modalities)]
        labels = batch[-1].to(device, dtype=torch.long)
        B = modals[0].size(0)

        with torch.no_grad():
            # 1. 预计算所有源模态的内容和风格编码 (全批次)
            all_masks, all_contents, all_styles = [], [], []
            for modal in modals:
                p_modal = mae.patchify(modal)
                all_masks.append((p_modal.var(dim=-1) > 0).float())
                embed = torch.cat([mae.cls_token.expand(B, -1, -1), mae.patch_embed(modal)], dim=1) + mae.pos_embed
                style_mean, style_std = mae.compute_style_mean_std(embed)
                all_styles.append((style_mean, style_std))
                z_c = embed
                for blk in mae.content_encoder: z_c = blk(z_c)
                all_contents.append(z_c)

            # [MODIFIED] 为当前批次中的每个样本初始化错误字典
            batch_mix_errors = [{} for _ in range(B)]
            batch_abs_errors = [{} for _ in range(B)]
            batch_incon_maps = [{} for _ in range(B)] 
            batch_labels = [labels[b].squeeze().cpu().numpy() for b in range(B)]

            # 2. 对所有翻译组合进行批次化推理
            for src_idx in range(n_modalities):
                src_modality_name = modality_names[src_idx]
                for tgt_modality_name in brats_modalities_list:
                    if src_modality_name == tgt_modality_name:
                        continue
                    
                    source_img, source_edge = modals[src_idx], edges[src_idx]
                    mask_source, (source_style_mean, source_style_std), source_content = all_masks[src_idx], all_styles[src_idx], all_contents[src_idx]
                    
                    # 2.1 源 -> 目标 (全批次)
                    z_s_dec = mae.decoder_embed(source_content)
                    cls_token, patch_tokens = z_s_dec[:, :1, :], z_s_dec[:, 1:, :]
                    edge_latent = mae.decoder_embed(mae.patch_embed_edge(source_edge))
                    fused_patch = mae.edge_fuse(torch.cat([patch_tokens, edge_latent], dim=-1))
                    fused_c = torch.cat([cls_token, fused_patch], dim=1) + mae.decoder_pos_embed[:, :z_s_dec.size(1), :]
                    target_style_mean = brats_style_features[tgt_modality_name]['mean']
                    target_style_std = brats_style_features[tgt_modality_name]['std']
                    z_s_to_t = mae.adain(fused_c, target_style_mean.expand(B, -1), target_style_std.expand(B, -1))
                    target_gen = mae.unpatchify(mae.generator(z_s_to_t)[:, 1:, :])
                    
                    # 2.2 目标 -> 源 (全批次)
                    embed_target_gen = torch.cat([mae.cls_token.expand(B, -1, -1), mae.patch_embed(target_gen)], dim=1) + mae.pos_embed
                    z_target_c = embed_target_gen
                    for blk in mae.content_encoder: z_target_c = blk(z_target_c)
                    
                    z_t_dec = mae.decoder_embed(z_target_c)
                    cls_token_t, patch_tokens_t = z_t_dec[:, :1, :], z_t_dec[:, 1:, :]
                    fused_patch_t = mae.edge_fuse(torch.cat([patch_tokens_t, edge_latent], dim=-1))
                    fused_c_t = torch.cat([cls_token_t, fused_patch_t], dim=1) + mae.decoder_pos_embed[:, :z_t_dec.size(1), :]
                    z_t_to_s = mae.adain(fused_c_t, source_style_mean, source_style_std)
                    source_recon = mae.unpatchify(mae.generator(z_t_to_s)[:, 1:, :])
                    
                    # 2.3 计算误差 (全批次)
                    abs_err_batch = (source_recon - source_img).abs().cpu().numpy()
                    
                    z_original = source_content[:, 1:, :]
                    z_recon = z_target_c[:, 1:, :]
                    sim_mat = content_cosine_similarity(z_original, z_recon, mask_source, mask_source)
                    diag_map_batch = get_diagonal_map(sim_mat).cpu().numpy()
                    
                    # 3. 将批次结果分离并存储到字典中
                    translation_key = f"{src_modality_name}_to_{tgt_modality_name}"
                    for b_idx in range(B):
                        abs_err = abs_err_batch[b_idx].squeeze()
                        diag_map = diag_map_batch[b_idx]
                        up_diag_map = cv2.resize(diag_map, (dataset_config['img_size'], dataset_config['img_size']), interpolation=cv2.INTER_LINEAR)
                        
                        incon_map = (1 - up_diag_map)
                        err_mix = abs_err * incon_map
                        
                        batch_mix_errors[b_idx][translation_key] = err_mix
                        batch_abs_errors[b_idx][translation_key] = abs_err
                        batch_incon_maps[b_idx][translation_key] = incon_map

            # 4. 异步保存每个样本的结果
            for b_idx in range(B):
                global_idx = i * opt.batch_size + b_idx + 1
                
                batch_mix_errors[b_idx]['label'] = batch_labels[b_idx]
                batch_abs_errors[b_idx]['label'] = batch_labels[b_idx]
                batch_incon_maps[b_idx]['label'] = batch_labels[b_idx]
                
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_mix_err.npz"), **batch_mix_errors[b_idx])
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_err.npz"), **batch_abs_errors[b_idx])
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_incon_map.npz"), **batch_incon_maps[b_idx])

    executor.shutdown(wait=True)
    print(f"--- Cycle Inference complete for {dataset_config['name']} ---")

# ==============================================================================
# PART 2: EVALUATION LOGIC (与 `final_evaluation_1.py` 完全相同)
# ==============================================================================
def parallel_median_filter(predictions_cpu, median_filter_size):
    if median_filter_size <= 1: return predictions_cpu
    with Pool(cpu_count()) as pool:
        tasks = [(predictions_cpu[i], median_filter_size) for i in range(predictions_cpu.shape[0])]
        smoothed_list = pool.starmap(ndimage.median_filter, tasks)
    return np.stack(smoothed_list)

def compute_auroc(p, t): return roc_auc_score(t.flatten(), p.flatten())
def compute_aupr(p, t):
    precision, recall, _ = precision_recall_curve(t.flatten(), p.flatten())
    return auc(recall, precision)

def apply_connected_component_analysis(binary_map, size):
    binary_map = binary_map.squeeze().astype(np.uint8)
    if size == 0: return torch.from_numpy(binary_map).unsqueeze(0)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    if num_labels <= 1: return torch.zeros_like(torch.from_numpy(binary_map)).unsqueeze(0)
    valid_labels = np.where(stats[1:, cv2.CC_STAT_AREA] >= size)[0] + 1
    return torch.from_numpy(np.isin(labels, valid_labels).astype(np.uint8)).unsqueeze(0)

def compute_dice(pred_bin, targets):
    pred_flat, targ_flat = pred_bin.view(-1).float(), targets.to(pred_bin.device).view(-1).float()
    intersection = (pred_flat * targ_flat).sum()
    return (2 * intersection) / (pred_flat.sum() + targ_flat.sum() + 1e-8)

def compute_dice_5_fold_cv(preds, targs, cc_size, n_thresh=100, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_dices = []
    indices = np.arange(preds.shape[0])
    for train_idx, test_idx in kf.split(indices):
        train_preds, train_targets = preds[train_idx], targs[train_idx]
        test_preds, test_targets = preds[test_idx], targs[test_idx]
        tmin, tmax = train_preds.min(), train_preds.max()
        best_thresh, best_dice_train = tmin, -1.0
        for t in torch.linspace(tmin, tmax, n_thresh):
            pred_bin = (train_preds > t).float()
            proc_bin = torch.stack([apply_connected_component_analysis(b.cpu().numpy(), cc_size) for b in pred_bin]).to(pred_bin.device)
            dice = compute_dice(proc_bin, train_targets)
            if dice > best_dice_train: best_dice_train, best_thresh = dice, t
        test_bin = (test_preds > best_thresh).float()
        proc_test_bin = torch.stack([apply_connected_component_analysis(b.cpu().numpy(), cc_size) for b in test_bin]).to(test_bin.device)
        dice_on_test = compute_dice(proc_test_bin, test_targets)
        fold_dices.append(dice_on_test.item())
    return np.mean(fold_dices), np.std(fold_dices)

# [NEW] 寻找最佳阈值（用于计算样本Dice）
def find_best_threshold(preds, targs, cc_size, n_thresh=100):
    tmin, tmax = preds.min(), preds.max()
    best_thresh, best_dice = tmin, -1.0
    for t in torch.linspace(tmin, tmax, n_thresh):
        pred_bin = (preds > t).float()
        proc_bin = torch.stack([apply_connected_component_analysis(b.cpu().numpy(), cc_size) for b in pred_bin]).to(pred_bin.device)
        dice = compute_dice(proc_bin, targs)
        if dice > best_dice:
            best_dice, best_thresh = dice, t
    return best_thresh, best_dice

# [NEW] 计算每个样本的Dice
def calculate_per_sample_dice(preds, targs, threshold, cc_size):
    pred_bin = (preds > threshold).float()
    proc_bin = torch.stack([apply_connected_component_analysis(b.cpu().numpy(), cc_size) for b in pred_bin]).to(preds.device)
    sample_dices = []
    for i in range(preds.shape[0]):
        dice = compute_dice(proc_bin[i], targs[i])
        sample_dices.append(dice.item())
    return sample_dices

# [MODIFIED] 修改 load_and_stack_data 以返回文件名
def load_and_stack_data(data_dir, fixed_size, pattern):
    files = sorted([f for f in os.listdir(data_dir) if pattern.match(f)])
    if not files: return None, None, None
    ann_dict, scores_dict = {}, {}
    sample_basenames = [] # [NEW]
    
    for fname in files:
        base, data = os.path.splitext(fname)[0], np.load(os.path.join(data_dir, fname))
        if 'label' not in data: continue
        
        label_tensor = F.interpolate(torch.from_numpy((data['label'] != 0).astype(np.int8)).unsqueeze(0).unsqueeze(0).float(), size=(fixed_size, fixed_size), mode="nearest").squeeze(0).long()
        ann_dict[base] = label_tensor
        sample_basenames.append(base) # [NEW]
        
        for key in [k for k in data.files if k != 'label']:
            score_tensor = F.interpolate(torch.from_numpy(data[key]).unsqueeze(0).unsqueeze(0).float(), size=(fixed_size, fixed_size), mode="bilinear", align_corners=False).squeeze(0)
            if key not in scores_dict: scores_dict[key] = {}
            scores_dict[key][base] = score_tensor
            
    if not ann_dict: return None, None, None
    
    targets, stacked_scores = torch.stack(list(ann_dict.values())), {}
    for method, scores in scores_dict.items():
        stacked_scores[method] = torch.stack([scores[base] for base in sample_basenames])
        
    return targets, stacked_scores, sample_basenames # [MODIFIED]

def cleanup_inference_results(output_path):
    if os.path.exists(output_path):
        print(f"Cleaning up inference results at: {output_path}")
        shutil.rmtree(output_path)

# ==============================================================================
# PART 3: MAIN EVALUATION SCRIPT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate model on validation sets using tuned CYCLE parameters.")
    parser.add_argument("--tuning_results_json", type=str, required=True, help="Path to the tuning_results_cycle.json file.")
    parser.add_argument("--results_dir", type=str, default="./final_evaluation_cycle_results", help="Directory to save final evaluation reports.")
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device to use.")
    # [NEW] 增加新参数
    parser.add_argument("--save_sample_dice",type=bool, default=False, help="Save per-sample Dice scores to an Excel file.")
    args = parser.parse_args()

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    
    with open(args.tuning_results_json, "r") as f:
        tuning_results = json.load(f)
    best_model_path = tuning_results["best_model_path"]
    log_file_path = os.path.join(args.results_dir, "final_evaluation_cycle_report.txt")

    # [NEW] 初始化 Excel writer (使用不同于 '1' 的文件名)
    excel_writer = None
    if args.save_sample_dice:
        excel_path = os.path.join(args.results_dir, "final_sample_dice_cycle_report.xlsx")
        excel_writer = pd.ExcelWriter(excel_path, engine='openpyxl')
        print(f"Per-sample Dice scores will be saved to: {excel_path}")

    opt = Namespace(
        img_size=256, dim_encoder=128, depth=12, num_heads=16, batch_size=8,
        dim_decoder=64, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.0, patch_size=8
    )

    def log(message):
        print(message)
        with open(log_file_path, "a") as f:
            f.write(message + "\n")

    log("="*80 + "\n" + " " * 16 + "FINAL CYCLE MODEL EVALUATION ON VALIDATION SETS" + "\n" + "="*80)
    log(f"Using best model: {os.path.basename(best_model_path)}")

    datasets = {
        'BraTS2021': {'modality': ['t1', 't2', 't1ce', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/test_edge','train_data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal', 'train_edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal_edge', 'fixed_size': 64, 'img_size': 256},
        'MSLUB': {'modality': ['t1', 't2','flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/train_edge', 'fixed_size': 256, 'img_size': 256},
        'MSSEG2015': {'modality': ['t2', 'flair', 'pd'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSSEG2015/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSSEG2015/train_edge', 'fixed_size': 256, 'img_size': 256},
        'ISLES2022': {'modality': ['adc', 'dwi', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/train_edge', 'fixed_size': 128, 'img_size': 256},
        'WMH': {'modality': ['t1', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/train', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/train_edge', 'fixed_size': 128, 'img_size': 256},
    }
    
    
    for name, config in datasets.items():
        config['name'], config['output_path'] = name, os.path.join(args.results_dir, f"inference_{name}_val")

    brats_style_features = compute_train_style_stats(datasets['BraTS2021'], best_model_path, device, opt)

    for name, config in datasets.items():
        log("\n" + "-"*60 + f"\nEVALUATING DATASET: {name}\n" + "-"*60)
        
        if name not in tuning_results["datasets"]:
            log(f"WARNING: No tuning results found for {name}. Skipping.")
            continue
        
        dataset_params = tuning_results["datasets"][name]
        mf_size, cc_size = dataset_params["best_median_filter_size"], dataset_params["best_connected_component_size"]
        log(f"Applying tuned hyperparameters: median_filter_size={mf_size}, connected_component_size={cc_size}")

        run_inference_for_dataset(best_model_path, config, brats_style_features, device, opt)

        sample_dice_data = [] # [NEW] 为当前数据集初始化列表

        for mode_suffix in ["_mix_err", "_err"]:
            mode = mode_suffix.strip('_')
            log(f"\n--- Evaluating Mode: '{mode}' ---")
            
            targets, stacked_scores, sample_basenames = load_and_stack_data(
                config['output_path'], config['fixed_size'], re.compile(fr'^\d+{mode_suffix}\.npz$')
            )
            
            if targets is None: log(f"No data found for mode '{mode}'. Skipping."); continue

            for seq_name, predictions in stacked_scores.items():
                if seq_name != 'label':
                    log(f"  Sequence: {seq_name}")
                    
                    filtered_np = parallel_median_filter(predictions.cpu().numpy().squeeze(), mf_size)
                    filtered_preds = torch.from_numpy(filtered_np).unsqueeze(1).to(device)

                    # 5-Fold CV Dice (用于报告)
                    dice_mean, dice_std = compute_dice_5_fold_cv(filtered_preds, targets, cc_size=cc_size)
                    log(f"    - 5-Fold CV Dice: {dice_mean:.4f} ± {dice_std:.4f}")
                    
                    # AUROC / AUPRC
                    auroc = compute_auroc(filtered_preds.cpu().numpy(), targets.cpu().numpy())
                    auprc = compute_aupr(filtered_preds.cpu().numpy(), targets.cpu().numpy())
                    log(f"    - AUROC         : {auroc:.4f}")
                    log(f"    - AUPRC         : {auprc:.4f}")
                    
                    # [NEW] 计算并保存每个样本的Dice
                    if args.save_sample_dice:
                        # 1. 找到在整个数据集上的最佳阈值
                        best_thresh, _ = find_best_threshold(filtered_preds, targets, cc_size=cc_size)
                        # 2. 使用该阈值计算每个样本的Dice
                        sample_dices = calculate_per_sample_dice(filtered_preds, targets, best_thresh, cc_size=cc_size)
                        # 3. 存储结果
                        for basename, dice_score in zip(sample_basenames, sample_dices):
                            sample_dice_data.append({
                                'filename': basename.replace(mode_suffix, ''), # 使用纯数字ID
                                'mode': mode,
                                'sequence': seq_name,
                                'dice_score': dice_score
                            })
                            
        # [NEW] 在数据集评估结束后，保存该数据集的 Excel sheet
        if args.save_sample_dice and sample_dice_data:
            df = pd.DataFrame(sample_dice_data)
            df_sorted = df.sort_values(by='dice_score', ascending=False)
            df_sorted.to_excel(excel_writer, sheet_name=name, index=False)
            log(f"Per-sample Dice scores for {name} saved to Excel sheet.")

        cleanup_inference_results(config['output_path'])
        
    # [NEW] 评估完成后，关闭 Excel writer
    if excel_writer:
        excel_writer.close()
        
    log("\n" + "="*80 + "\nALL VALIDATION SETS EVALUATED. Report saved to final_evaluation_cycle_report.txt\n" + "="*80)

if __name__ == "__main__":
    main()
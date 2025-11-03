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
import cv2
from scipy import ndimage
from argparse import Namespace
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils.MM_loader import get_maeloader
from model.MMPC_any import MultiModalPatchMAE

def content_cosine_similarity(zA, zB, maskA=None, maskB=None):
    B, L, D = zA.shape
    sim_matrix = F.cosine_similarity(zA.unsqueeze(2), zB.unsqueeze(1), dim=-1)
    if maskA is not None and maskB is not None:
        valid_2d = (maskA.unsqueeze(2) * maskB.unsqueeze(1)) > 0
    else:
        valid_2d = torch.ones_like(sim_matrix, dtype=torch.bool)
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
    all_means, all_stds = [[] for _ in range(n_modalities)], [[] for _ in range(n_modalities)]
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Computing style stats", leave=False):
            modals = [m.to(device, dtype=torch.float) for m in batch[:n_modalities]]
            for i, modal in enumerate(modals):
                embed = torch.cat([mae.cls_token.expand(modal.shape[0], -1, -1), mae.patch_embed(modal)], dim=1) + mae.pos_embed
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

# [MODIFIED] run_inference_for_dataset 函数，采用高效的全批次处理
def run_inference_for_dataset(model_path, dataset_config, brats_style_features, device, opt):
    print(f"\n--- Running CYCLE inference for dataset: {dataset_config['name']} with model: {os.path.basename(model_path)} ---")
    
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

    test_loader = get_maeloader(
        batchsize=opt.batch_size, shuffle=False, pin_memory=True, img_size=dataset_config['img_size'],
        img_root=dataset_config['data_root'], num_workers=32, augment=False, if_addlabel=True,
        modality=dataset_config['modality'], edge_root=dataset_config['edge_root']
    )
    
    modality_names = {idx: name for idx, name in enumerate(dataset_config['modality'])}
    n_modalities = len(dataset_config['modality'])
    brats_modalities_list = ['t1', 't1ce', 't2', 'flair']

    for i, batch in enumerate(tqdm(test_loader, desc=f"Cycle Inference on {dataset_config['name']}", leave=False)):
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

            # [NEW] 为当前批次中的每个样本初始化错误字典
            batch_mix_errors = [{'label': labels[b].squeeze().cpu().numpy()} for b in range(B)]
            batch_abs_errors = [{'label': labels[b].squeeze().cpu().numpy()} for b in range(B)]

            # 2. 对所有翻译组合进行批次化推理
            for src_idx in range(n_modalities):
                src_modality_name = modality_names[src_idx]
                for tgt_modality_name in brats_modalities_list:
                    if src_modality_name == tgt_modality_name: continue
                    
                    # 提取当前批次的源数据
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
                    abs_err_batch = (source_recon - source_img).abs()
                    z_original, z_recon = source_content[:, 1:, :], z_target_c[:, 1:, :]
                    sim_mat = content_cosine_similarity(z_original, z_recon, mask_source, mask_source)
                    diag_map_batch = get_diagonal_map(sim_mat).unsqueeze(1) # (B, 1, H_p, W_p)
                    
                    up_diag_map_batch = F.interpolate(
                        diag_map_batch,
                        size=(dataset_config['img_size'], dataset_config['img_size']),
                        mode='bilinear',
                        align_corners=False
                    ) # (B, 1, H, W)
                    err_mix_batch = abs_err_batch * (1 - up_diag_map_batch)

                    # [NEW] 3. 将批次结果分离并存储到字典中
                    translation_key = f"{src_modality_name}_to_{tgt_modality_name}"
                    for b_idx in range(B):
                        batch_mix_errors[b_idx][translation_key] = err_mix_batch[b_idx].squeeze().cpu().numpy()
                        batch_abs_errors[b_idx][translation_key] = abs_err_batch[b_idx].squeeze().cpu().numpy()

            # [NEW] 4. 异步保存每个样本的结果
            for b_idx in range(B):
                global_idx = i * opt.batch_size + b_idx + 1
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_mix_err.npz"), **batch_mix_errors[b_idx])
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_err.npz"), **batch_abs_errors[b_idx])

    executor.shutdown(wait=True)
    print(f"--- Cycle Inference complete for {dataset_config['name']} ---")


# ==============================================================================
# PART 2: EVALUATION LOGIC (与原版相同)
# ... (这部分代码与原版相同，为了简洁性在此省略，实际使用时请保留)
# ==============================================================================
def parallel_median_filter(predictions_cpu, median_filter_size):
    if median_filter_size <= 1: return predictions_cpu
    with Pool(cpu_count()) as pool:
        tasks = [(predictions_cpu[i], median_filter_size) for i in range(predictions_cpu.shape[0])]
        smoothed_list = pool.starmap(ndimage.median_filter, tasks)
    return np.stack(smoothed_list)

def apply_connected_component_analysis(binary_map, connected_component_size=10):
    binary_map = binary_map.squeeze().astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    if num_labels <= 1 or connected_component_size == 0: return torch.from_numpy(binary_map).unsqueeze(0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid_labels = np.where(areas >= connected_component_size)[0] + 1
    processed_map = np.isin(labels, valid_labels).astype(np.uint8)
    return torch.from_numpy(processed_map).unsqueeze(0)

def compute_dice(pred_bin, targets):
    pred_flat, targ_flat = pred_bin.view(-1).float(), targets.to(pred_bin.device).view(-1).float()
    intersection = (pred_flat * targ_flat).sum()
    dice = (2 * intersection) / (pred_flat.sum() + targ_flat.sum() + 1e-8)
    # [MODIFICATION] Call .item() on the final result to convert Tensor to float
    return dice.item()

def compute_best_dice_for_sequence(predictions, targets, connected_component_size, n_thresh=100):
    tmin, tmax = predictions.min().item(), predictions.max().item()
    best_dice = 0.0
    for t in np.linspace(tmin, tmax, n_thresh):
        pred_bin = (predictions > t).float()
        processed_bin = torch.stack([apply_connected_component_analysis(p.cpu().numpy(), connected_component_size) for p in pred_bin]).to(pred_bin.device)
        dice = compute_dice(processed_bin, targets)
        if dice > best_dice: best_dice = dice
    return best_dice

def load_and_stack_data(data_dir, fixed_size, pattern):
    files = sorted([f for f in os.listdir(data_dir) if pattern.match(f)])
    if not files: return None, None
    ann_dict, scores_dict = {}, {}
    for fname in files:
        base, data = os.path.splitext(fname)[0], np.load(os.path.join(data_dir, fname))
        if 'label' not in data: continue
        label_tensor = F.interpolate(torch.from_numpy((data['label'] != 0).astype(np.int8)).unsqueeze(0).unsqueeze(0).float(), size=(fixed_size, fixed_size), mode="nearest").squeeze(0).long()
        ann_dict[base] = label_tensor
        for key in [k for k in data.files if k != 'label']:
            score_tensor = F.interpolate(torch.from_numpy(data[key]).unsqueeze(0).unsqueeze(0).float(), size=(fixed_size, fixed_size), mode="bilinear", align_corners=False).squeeze(0)
            if key not in scores_dict: scores_dict[key] = {}
            scores_dict[key][base] = score_tensor
    if not ann_dict: return None, None
    targets = torch.stack(list(ann_dict.values()))
    stacked_scores = {method: torch.stack(list(scores.values())) for method, scores in scores_dict.items()}
    return targets, stacked_scores

def cleanup_inference_results(output_path):
    if os.path.exists(output_path):
        print(f"Cleaning up inference results at: {output_path}")
        shutil.rmtree(output_path)

# ==============================================================================
# PART 3: MAIN TUNING SCRIPT (与原版相同)
# ... (这部分代码与原版相同，为了简洁性在此省略，实际使用时请保留)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Find best model and hyperparameters using CYCLE TRANSLATION.")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing model checkpoints (.pth files).")
    parser.add_argument("--results_dir", type=str, default="./tuning_results_cycle", help="Directory to save tuning results.")
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device to use.")
    args = parser.parse_args()

    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    log_file_path = os.path.join(args.results_dir, "tuning_log_cycle.txt")
    
    opt = Namespace(
        img_size=256, dim_encoder=128, depth=12, num_heads=16, batch_size=8,
        dim_decoder=64, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.0, patch_size=8
    )

    def log(message):
        print(message)
        with open(log_file_path, "a") as f: f.write(message + "\n")

    datasets = {
        'BraTS2021': {'modality': ['t1', 't1ce', 't2', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/eval', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/eval_edge', 'train_data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal', 'train_edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal_edge', 'fixed_size': 64, 'img_size': 256},
        'MSLUB': {'modality': ['t1', 't2','flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/test_edge', 'fixed_size': 256, 'img_size': 256},
        'MSSEG2015': {'modality': ['t2', 'flair', 'pd'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSSEG2015/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSSEG2015/test_edge', 'fixed_size': 256, 'img_size': 256},
        'ISLES2022': {'modality': ['adc', 'dwi', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/test_edge', 'fixed_size': 128, 'img_size': 256},
        'WMH': {'modality': ['t1', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/test_edge', 'fixed_size': 128, 'img_size': 256},
    }
    
    
    for name, config in datasets.items():
        config['name'] = name
        config['output_path'] = os.path.join(args.results_dir, f"inference_{name}")

    log("\n" + "="*80 + "\nSTEP 1: Searching for the best model checkpoint (CYCLE method)..." + "\n" + "="*80)
    model_files = sorted([f for f in os.listdir(args.models_dir) if f.endswith(".pth")])
    best_model_path, best_model_avg_dice, best_sequences_per_dataset = None, -1.0, {}

    for model_file in model_files:
        model_path = os.path.join(args.models_dir, model_file)
        log(f"\nEvaluating Model: {model_file}")
        brats_style_features = compute_train_style_stats(datasets['BraTS2021'], model_path, device, opt)
        current_model_dataset_dices, current_model_best_sequences = [], {}
        for name, config in datasets.items():
            run_inference_for_dataset(model_path, config, brats_style_features, device, opt)
            targets, stacked_scores = load_and_stack_data(config['output_path'], config['fixed_size'], re.compile(r'^\d+_mix_err\.npz$'))
            if targets is None: log(f"  - WARNING: No data for {name}. Skipping."); continue
            max_dice, best_seq = -1.0, None
            for seq_name, predictions in stacked_scores.items():
                dice = compute_best_dice_for_sequence(predictions, targets, connected_component_size=0)
                if dice > max_dice: max_dice, best_seq = dice, seq_name
            log(f"  - Dataset {name}: Best Sequence='{best_seq}', Dice={max_dice:.4f}")
            current_model_dataset_dices.append(max_dice)
            current_model_best_sequences[name] = best_seq
            cleanup_inference_results(config['output_path'])

        if current_model_dataset_dices:
            avg_dice = np.mean(current_model_dataset_dices)
            log(f"Model {model_file} Average Dice: {avg_dice:.4f}")
            if avg_dice > best_model_avg_dice:
                best_model_avg_dice, best_model_path, best_sequences_per_dataset = avg_dice, model_path, current_model_best_sequences
                log(f"!!! New best model found: {os.path.basename(best_model_path)} (Avg Dice: {best_model_avg_dice:.4f}) !!!")

    log("\n" + "="*80 + f"\nSTEP 1 COMPLETE. Best Model: {os.path.basename(best_model_path)}\n" + f"Best recorded sequences: {json.dumps(best_sequences_per_dataset, indent=2)}\n" + "="*80)

    log("\n" + "="*80 + "\nSTEP 2: Searching for best hyperparameters (CYCLE method)..." + f"\nUsing model: {os.path.basename(best_model_path)}\n" + "="*80)
    final_tuning_results = {"best_model_path": best_model_path, "datasets": {}}
    best_model_brats_styles = compute_train_style_stats(datasets['BraTS2021'], best_model_path, device, opt)
    for name, config in datasets.items():
        log(f"\n--- Tuning for dataset: {name} ---")
        best_sequence = best_sequences_per_dataset.get(name)
        if not best_sequence: log(f"  - No best sequence for {name}. Skipping."); continue
        log(f"  - Using best sequence: {best_sequence}")
        run_inference_for_dataset(best_model_path, config, best_model_brats_styles, device, opt)
        targets, stacked_scores = load_and_stack_data(config['output_path'], config['fixed_size'], re.compile(r'^\d+_mix_err\.npz$'))
        if targets is None or best_sequence not in stacked_scores:
            log(f"  - ERROR: Could not load data for sequence '{best_sequence}'. Skipping."); cleanup_inference_results(config['output_path']); continue
        predictions, predictions_cpu = stacked_scores[best_sequence], stacked_scores[best_sequence].cpu().numpy()
        log("  - Searching for best median_filter_size (cc_size=0)...")
        median_filter_sizes = range(0, 10, 3)
        base_dice = compute_best_dice_for_sequence(predictions, targets, 0)
        log(f"    - median_filter_size=0 (baseline), Dice={base_dice:.4f}")
        best_mf_dice, best_mf_size = base_dice, 0
        for mf_size in median_filter_sizes:
            if mf_size == 0: continue
            filtered_preds = torch.from_numpy(parallel_median_filter(predictions_cpu.squeeze(), mf_size)).unsqueeze(1).to(device)
            dice = compute_best_dice_for_sequence(filtered_preds, targets, 0)
            log(f"    - median_filter_size={mf_size}, Dice={dice:.4f}")
            if dice > best_mf_dice: best_mf_dice, best_mf_size = dice, mf_size
        log(f"  - Best median_filter_size: {best_mf_size} (Dice: {best_mf_dice:.4f})")

        log(f"  - Searching for best connected_component_size (median_filter_size={best_mf_size})...")
        cc_sizes = range(0, 61, 10)
        best_cc_dice, best_cc_size = best_mf_dice, 0
        final_filtered_preds = torch.from_numpy(parallel_median_filter(predictions_cpu.squeeze(), best_mf_size)).unsqueeze(1).to(device) if best_mf_size > 0 else predictions
        for cc_size in cc_sizes:
            if cc_size == 0: continue
            dice = compute_best_dice_for_sequence(final_filtered_preds, targets, cc_size)
            log(f"    - connected_component_size={cc_size}, Dice={dice:.4f}")
            if dice > best_cc_dice: best_cc_dice, best_cc_size = dice, cc_size
        log(f"  - Best connected_component_size: {best_cc_size} (Dice: {best_cc_dice:.4f})")
        final_tuning_results["datasets"][name] = {"best_sequence": best_sequence, "best_median_filter_size": best_mf_size, "best_connected_component_size": best_cc_size, "final_tuned_dice": best_cc_dice}
        cleanup_inference_results(config['output_path'])

    results_file_path = os.path.join(args.results_dir, "tuning_results_cycle.json")
    with open(results_file_path, "w") as f: json.dump(final_tuning_results, f, indent=4)
    log("\n" + "="*80 + f"\nTUNING COMPLETE! Final results saved to: {results_file_path}\n" + "="*80)
    log(json.dumps(final_tuning_results, indent=4))

if __name__ == "__main__":
    main()
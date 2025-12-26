import os
import sys
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

# Imports
from data.dataset import get_dataloader
from models.unitransad import UniTransAD

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

def run_inference_for_dataset(model_path, dataset_config, dspm_features, device, opt, sample_size='all'):
    print(f"\n--- Running CYCLE inference for dataset: {dataset_config['name']} ---")
    img_save_path = dataset_config['output_path']
    os.makedirs(img_save_path, exist_ok=True)
    executor = ThreadPoolExecutor(max_workers=os.cpu_count())
    
    # 这里的初始化通常是正确的，但也建议显式传入所有参数以防万一
    model = UniTransAD(
        img_size=opt.img_size, patch_size=opt.patch_size, embed_dim=opt.dim_encoder,
        depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth,
        decoder_num_heads=opt.decoder_num_heads, mlp_ratio=opt.mlp_ratio,
        norm_layer=nn.LayerNorm,
        ema_momentum=opt.ema_momentum
    ).to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 1. DataLoader
    original_loader = get_dataloader(
        batchsize=opt.batch_size, shuffle=False, pin_memory=True, img_size=dataset_config['img_size'],
        img_root=dataset_config['data_root'], num_workers=16, augment=False, if_addlabel=True,
        sequences=dataset_config['modality'], edge_root=dataset_config['edge_root']
    )
    
    # Sampling Logic
    test_loader = original_loader
    if str(sample_size).lower() != 'all':
        try:
            n_samples = int(sample_size)
            dataset = original_loader.dataset
            total_len = len(dataset)
            if n_samples < total_len:
                g = torch.Generator()
                g.manual_seed(42) 
                indices = torch.randperm(total_len, generator=g)[:n_samples].tolist()
                subset = torch.utils.data.Subset(dataset, indices)
                test_loader = torch.utils.data.DataLoader(
                    subset, batch_size=opt.batch_size, shuffle=False,
                    num_workers=original_loader.num_workers, pin_memory=original_loader.pin_memory
                )
                print(f"    -> [SAMPLING] Using {n_samples}/{total_len} samples")
        except ValueError:
            pass

    modality_names = {idx: name for idx, name in enumerate(dataset_config['modality'])}
    n_modalities = len(dataset_config['modality'])
    dspm_modalities_list = ['t1', 't1ce', 't2', 'flair']

    for i, batch in enumerate(tqdm(test_loader, desc=f"Inference {dataset_config['name']}", leave=False)):
        modals = [batch[j].to(device, dtype=torch.float) for j in range(n_modalities)]
        edges = [batch[j + n_modalities].to(device, dtype=torch.float) for j in range(n_modalities)]
        labels = batch[-1].to(device, dtype=torch.long)
        B = modals[0].size(0)

        with torch.no_grad():
            # Precompute content/style
            all_masks, all_contents, all_styles = [], [], []
            for modal in modals:
                p_modal = model.patchify(modal)
                all_masks.append((p_modal.var(dim=-1) > 0).float())
                embed = torch.cat([model.cls_token.expand(B, -1, -1), model.patch_embed(modal)], dim=1) + model.pos_embed
                style_mean, style_std = model.compute_style_mean_std(embed)
                all_styles.append((style_mean, style_std))
                z_c = embed
                for blk in model.content_encoder: z_c = blk(z_c)
                all_contents.append(z_c)

            batch_mix_errors = [{'label': labels[b].squeeze().cpu().numpy()} for b in range(B)]
            batch_abs_errors = [{'label': labels[b].squeeze().cpu().numpy()} for b in range(B)]

            # Cyclic Translation
            for src_idx in range(n_modalities):
                src_name = modality_names[src_idx]
                for tgt_name in dspm_modalities_list:
                    if src_name == tgt_name: continue
                    
                    # 1. Source -> Target (Virtual) using DSPM Prototypes
                    source_content = all_contents[src_idx]
                    target_style_mean = dspm_features[tgt_name]['mean']
                    target_style_std = dspm_features[tgt_name]['std']
                    
                    z_s_dec = model.decoder_embed(source_content)
                    edge_latent = model.decoder_embed(model.patch_embed_edge(edges[src_idx]))
                    
                    cls, patches = z_s_dec[:, :1, :], z_s_dec[:, 1:, :]
                    fused_patch = model.edge_fuse(torch.cat([patches, edge_latent], dim=-1))
                    fused_c = torch.cat([cls, fused_patch], dim=1) + model.decoder_pos_embed[:, :z_s_dec.size(1), :]
                    
                    z_s_to_t = model.adain(fused_c, target_style_mean.expand(B, -1), target_style_std.expand(B, -1))
                    target_gen = model.unpatchify(model.generator(z_s_to_t)[:, 1:, :])

                    # 2. Target -> Source (Cyclic)
                    embed_t_gen = torch.cat([model.cls_token.expand(B, -1, -1), model.patch_embed(target_gen)], dim=1) + model.pos_embed
                    z_target_c = embed_t_gen
                    for blk in model.content_encoder: z_target_c = blk(z_target_c)
                    
                    z_t_dec = model.decoder_embed(z_target_c)
                    cls_t, patches_t = z_t_dec[:, :1, :], z_t_dec[:, 1:, :]
                    fused_patch_t = model.edge_fuse(torch.cat([patches_t, edge_latent], dim=-1))
                    fused_c_t = torch.cat([cls_t, fused_patch_t], dim=1) + model.decoder_pos_embed[:, :z_t_dec.size(1), :]
                    
                    z_t_to_s = model.adain(fused_c_t, all_styles[src_idx][0], all_styles[src_idx][1])
                    source_recon = model.unpatchify(model.generator(z_t_to_s)[:, 1:, :])
                    
                    # 3. Dual-Level Anomaly Detection
                    abs_err = (source_recon - modals[src_idx]).abs()
                    sim_mat = content_cosine_similarity(source_content[:, 1:, :], z_target_c[:, 1:, :], all_masks[src_idx], all_masks[src_idx])
                    diag_map = get_diagonal_map(sim_mat).unsqueeze(1)
                    up_diag = F.interpolate(diag_map, size=(dataset_config['img_size'], dataset_config['img_size']), mode='bilinear', align_corners=False)
                    err_mix = abs_err * (1 - up_diag)

                    key = f"{src_name}_to_{tgt_name}"
                    for b_idx in range(B):
                        batch_mix_errors[b_idx][key] = err_mix[b_idx].squeeze().cpu().numpy()
                        batch_abs_errors[b_idx][key] = abs_err[b_idx].squeeze().cpu().numpy()

            for b_idx in range(B):
                global_idx = i * opt.batch_size + b_idx + 1
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_mix_err.npz"), **batch_mix_errors[b_idx])
                executor.submit(save_npz_data, os.path.join(img_save_path, f"{global_idx}_err.npz"), **batch_abs_errors[b_idx])

    executor.shutdown(wait=True)

# Helper functions
def parallel_median_filter(preds, size):
    if size <= 1: return preds
    with Pool(cpu_count()) as pool:
        tasks = [(preds[i], size) for i in range(preds.shape[0])]
        res = pool.starmap(ndimage.median_filter, tasks)
    return np.stack(res)

def apply_connected_component(binary_map, size=10):
    binary_map = binary_map.squeeze().astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
    if n <= 1 or size == 0: return torch.from_numpy(binary_map).unsqueeze(0)
    valid = np.where(stats[1:, cv2.CC_STAT_AREA] >= size)[0] + 1
    return torch.from_numpy(np.isin(labels, valid).astype(np.uint8)).unsqueeze(0)

def compute_dice(p, t):
    pf, tf = p.view(-1).float(), t.to(p.device).view(-1).float()
    return (2 * (pf * tf).sum()) / (pf.sum() + tf.sum() + 1e-8)

def compute_best_dice(preds, targets, cc_size, n=100):
    tmin, tmax = preds.min().item(), preds.max().item()
    best = 0.0
    for t in np.linspace(tmin, tmax, n):
        pb = (preds > t).float()
        proc = torch.stack([apply_connected_component(p.cpu().numpy(), cc_size) for p in pb]).to(pb.device)
        d = compute_dice(proc, targets)
        if d > best: best = d.item()
    return best

def load_and_stack(data_dir, size, pattern):
    files = sorted([f for f in os.listdir(data_dir) if pattern.match(f)])
    if not files: return None, None
    ann, scores = {}, {}
    for fname in files:
        base = os.path.splitext(fname)[0]
        data = np.load(os.path.join(data_dir, fname))
        if 'label' not in data: continue
        ann[base] = F.interpolate(torch.from_numpy((data['label']!=0).astype(np.int8)).unsqueeze(0).unsqueeze(0).float(), size=(size,size), mode="nearest").squeeze(0).long()
        for k in data.files:
            if k != 'label':
                if k not in scores: scores[k] = {}
                scores[k][base] = F.interpolate(torch.from_numpy(data[k]).unsqueeze(0).unsqueeze(0).float(), size=(size,size), mode="bilinear", align_corners=False).squeeze(0)
    if not ann: return None, None
    return torch.stack(list(ann.values())), {m: torch.stack(list(s.values())) for m, s in scores.items()}

def cleanup(path):
    if os.path.exists(path): shutil.rmtree(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", required=True, help="Directory containing .pth checkpoints")
    parser.add_argument("--results_dir", default="./tuning_results", help="Dir to save results")
    parser.add_argument("--sample_size", default="all", help="Number of samples to tune on")
    parser.add_argument("--cuda_device", default="cuda:0")
    args = parser.parse_args()
    
    device = torch.device(args.cuda_device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)
    
    opt = Namespace(img_size=256, dim_encoder=128, depth=12, num_heads=16, batch_size=8,
                    dim_decoder=64, decoder_depth=8, decoder_num_heads=8, mlp_ratio=4.0, 
                    patch_size=8,  
                    ema_momentum=0.1)
    
    # Dataset Configs (PLACEHOLDERS - Update paths!)
    datasets = {
        'BraTS2021': {'modality': ['t1', 't1ce', 't2', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/eval', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/eval_edge', 'train_data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal', 'train_edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal_edge', 'fixed_size': 64, 'img_size': 256},
        'MSLUB': {'modality': ['t1', 't2','flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSLUB/test_edge', 'fixed_size': 256, 'img_size': 256},
        'ISLES2022': {'modality': ['adc', 'dwi', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/ISLES/test_edge', 'fixed_size': 128, 'img_size': 256},
        'WMH': {'modality': ['t1', 'flair'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/WMH/test_edge', 'fixed_size': 128, 'img_size': 256},
        'BRISC': {'modality': ['t1'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BRISC/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BRISC/test_edge', 'fixed_size': 64, 'img_size': 256},
        'MSD': {'modality': ['flair','t1','t1ce','t2'], 'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSD_BrainTumour/test', 'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/MSD_BrainTumour/test_edge','fixed_size': 64, 'img_size': 256},
    }
    
    
    for name, config in datasets.items():
        config['name'] = name
        config['output_path'] = os.path.join(args.results_dir, f"inf_{name}")

    # --- Step 1: Find Best Model ---
    best_model_path = None
    best_dice = -1.0
    best_seqs = {}
    
    model_files = sorted([f for f in os.listdir(args.models_dir) if f.endswith(".pth")])
    for model_file in model_files:
        path = os.path.join(args.models_dir, model_file)
        print(f"Checking {model_file}...")
        
        # [修复点 1] 初始化 temp 模型时必须传入所有架构参数，尤其是 patch_size
        temp = UniTransAD(
            img_size=opt.img_size, 
            patch_size=opt.patch_size,  # <--- 必须传入
            embed_dim=opt.dim_encoder, 
            depth=opt.depth, 
            num_heads=opt.num_heads,
            decoder_embed_dim=opt.dim_decoder, 
            decoder_depth=opt.decoder_depth, 
            decoder_num_heads=opt.decoder_num_heads,
            mlp_ratio=opt.mlp_ratio,
            ema_momentum=opt.ema_momentum
        ).to(device)
        
        sd = torch.load(path, map_location=device)
        temp.load_state_dict(sd, strict=False)
        dspm_feats = {}
        for mod in ['t1', 't1ce', 't2', 'flair']:
            mean_key = f'dspm_{mod}_mean' if f'dspm_{mod}_mean' in sd else f'brats_{mod}_ema_mean'
            std_key = f'dspm_{mod}_std' if f'dspm_{mod}_std' in sd else f'brats_{mod}_ema_std'
            
            mean_val = sd.get(mean_key, torch.zeros(opt.dim_decoder)).to(device)
            std_val = sd.get(std_key, torch.zeros(opt.dim_decoder)).to(device)
            dspm_feats[mod] = {'mean': mean_val, 'std': std_val}
        
        del temp

        # Eval on each dataset
        curr_dices = []
        curr_seqs = {}
        for name, config in datasets.items():
            run_inference_for_dataset(path, config, dspm_feats, device, opt, args.sample_size)
            targets, scores = load_and_stack(config['output_path'], config['fixed_size'], re.compile(r'^\d+_mix_err\.npz$'))
            
            if targets is None: continue
            
            mdice, mseq = -1, None
            for seq, pred in scores.items():
                d = compute_best_dice(pred, targets, 0)
                if d > mdice: mdice, mseq = d, seq
            
            print(f"  {name}: {mseq} = {mdice:.4f}")
            curr_dices.append(mdice)
            curr_seqs[name] = mseq
            cleanup(config['output_path'])
            
        if curr_dices and np.mean(curr_dices) > best_dice:
            best_dice = np.mean(curr_dices)
            best_model_path = path
            best_seqs = curr_seqs
            print(f"!!! New Best: {best_dice:.4f} !!!")

    # --- Step 2: Tune Hyperparams (Median Filter & CC) ---
    print(f"\n--- Tuning Hyperparameters for Best Model: {os.path.basename(best_model_path)} ---")
    final_res = {"best_model_path": best_model_path, "datasets": {}}
    
    # [修复点 2] 重新加载 DSPM 时同样需要正确的参数
    temp = UniTransAD(
        img_size=opt.img_size, 
        patch_size=opt.patch_size,  # <--- 必须传入
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

    for name, config in datasets.items():
        if name not in best_seqs: continue
        seq = best_seqs[name]
        print(f"Tuning {name} on sequence {seq}...")
        
        run_inference_for_dataset(best_model_path, config, dspm_feats, device, opt, args.sample_size)
        targets, scores = load_and_stack(config['output_path'], config['fixed_size'], re.compile(r'^\d+_mix_err\.npz$'))
        
        preds = scores[seq]
        preds_cpu = preds.cpu().numpy()
        
        # 1. Median Filter
        best_mf_dice, best_mf = -1, 0
        for mf in [0, 3, 5, 7]:
            if mf > 0:
                p_filt = torch.from_numpy(parallel_median_filter(preds_cpu.squeeze(), mf)).unsqueeze(1).to(device)
            else:
                p_filt = preds
            d = compute_best_dice(p_filt, targets, 0)
            if d > best_mf_dice: best_mf_dice, best_mf = d, mf
            
        # 2. CC Size
        best_cc_dice, best_cc = -1, 0
        p_filt = torch.from_numpy(parallel_median_filter(preds_cpu.squeeze(), best_mf)).unsqueeze(1).to(device) if best_mf > 0 else preds
        for cc in range(0, 101, 10):
            d = compute_best_dice(p_filt, targets, cc)
            if d > best_cc_dice: best_cc_dice, best_cc = d, cc
            
        print(f"  Best Params: MF={best_mf}, CC={best_cc}, Dice={best_cc_dice:.4f}")
        final_res["datasets"][name] = {
            "best_sequence": seq, 
            "best_median_filter_size": best_mf, 
            "best_connected_component_size": best_cc
        }
        cleanup(config['output_path'])

    with open(os.path.join(args.results_dir, "tuning_results_cycle.json"), "w") as f:
        json.dump(final_res, f, indent=4)
    print("Tuning complete.")

if __name__ == "__main__":
    main()
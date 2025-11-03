import torch
import logging
import numpy as np
from utils.MM_loader import *
from model.MMPC_any import *
from utils.mae_visualize import *
from options import Pretrain_Options
import os
from tensorboardX import SummaryWriter
from torchvision.utils import save_image 
import itertools

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse

def setup_ddp():
    """初始化分布式进程组"""
    dist.init_process_group(backend="nccl") 
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    """销毁进程组"""
    dist.destroy_process_group()

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    setup_ddp()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    opt = Pretrain_Options().get_opt()

    datasets_config = [
        {
            'name': 'BraTS2021',
            'modality': ['t1', 't2', 't1ce', 'flair'],
            'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal',
            'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/BraTS2021/train_normal_edge'
        },
        {
            'name': 'IXI',
            'modality': ['t1', 't2', 'pd'],
            'data_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/IXI/train',
            'edge_root': '/mnt/sdb/zq/brain_sas_baseline/datasets/npy/IXI/train_edge'
        }
    ]

    mae = MultiModalPatchMAE(img_size=opt.img_size, patch_size=opt.patch_size, 
                             embed_dim=opt.dim_encoder, depth=opt.depth, 
                             num_heads=opt.num_heads, in_chans=1,
                             decoder_embed_dim=opt.dim_decoder, 
                             decoder_depth=opt.decoder_depth, 
                             decoder_num_heads=opt.decoder_num_heads,
                             mlp_ratio=opt.mlp_ratio, norm_layer=nn.LayerNorm)

    if rank == 0:
        os.makedirs(opt.img_save_path, exist_ok=True)
        os.makedirs(opt.weight_save_path, exist_ok=True)

    device = torch.device(f"cuda:{rank}")
    mae = mae.to(device)

    mae = DDP(mae, device_ids=[rank], find_unused_parameters=True)

    train_loaders = []
    train_samplers = []
    for dataset_config in datasets_config:
        from utils.MM_loader import MultiModalDataset 
        train_dataset = MultiModalDataset(
            image_root=dataset_config['data_root'],
            edge_root=dataset_config['edge_root'],
            img_size=opt.img_size,
            augment=opt.augment,
            modal=dataset_config['modality']
        )

        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        train_samplers.append(sampler)
        
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=False,  
            num_workers=opt.num_workers,
            pin_memory=True,
            sampler=sampler,
            drop_last=True 
        )
        train_loaders.append(loader)

    optimizer = torch.optim.Adam(mae.parameters(), lr=opt.lr, betas=(0.9, 0.95))

    if opt.use_checkpoints:
        print(f'Rank {rank} loading checkpoint......', opt.checkpoint_path)
        # DDP-MOD: 加载模型时，需要确保所有进程都从同一个文件加载
        dist.barrier() # 等待所有进程到达这里
        map_location = {'cuda:0': f'cuda:{rank}'}
        state_dict = torch.load(opt.checkpoint_path, map_location=map_location)
        mae.module.load_state_dict(state_dict) 

    writer = None
    if rank == 0:
        logging.basicConfig(filename=opt.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        writer = SummaryWriter(log_dir=opt.logs_path)

    for epoch in range(1, opt.epoch):
        for sampler in train_samplers:
            sampler.set_epoch(epoch)
            
        total_iterations = sum(len(loader) for loader in train_loaders)
        iterators = [iter(loader) for loader in train_loaders]
        dataset_indices = []
        for idx, loader in enumerate(train_loaders):
            dataset_indices.extend([idx] * len(loader))
        
        np.random.shuffle(dataset_indices)
        progress = {idx: 0 for idx in range(len(train_loaders))}
        
        for i, loader_idx in enumerate(dataset_indices):
            dataset_config = datasets_config[loader_idx]
            num_modalities = len(dataset_config['modality'])
            modality_names = dataset_config['modality']
            
            try:
                data = next(iterators[loader_idx])
                progress[loader_idx] += 1
            except StopIteration:
                continue 
            
            modalities = []
            edges = []
            half_idx = len(data) // 2
            for j in range(num_modalities):
                modalities.append(data[j].to(device, dtype=torch.float))
                edges.append(data[j + half_idx].to(device, dtype=torch.float))
            
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            optimizer.zero_grad()
            outputs = mae(modalities, edges)
            
            nce_loss = outputs['nce_loss']
            rec_losses = outputs['rec_losses']
            rec_images = outputs['rec_images']
            
            nce_loss = nce_loss * opt.nce_weight
            weighted_rec_losses = [loss * opt.rec_weight for loss in rec_losses]
            avg_rec_loss = sum(weighted_rec_losses) / len(weighted_rec_losses) * 5
            total_loss = nce_loss + avg_rec_loss
            
            total_loss.backward()
            optimizer.step()
            
            if rank == 0:
                global_step = epoch * total_iterations + i
                writer.add_scalar('Loss/total_loss', total_loss.item(), global_step)
                writer.add_scalar('Loss/nce_loss', nce_loss.item(), global_step)
                writer.add_scalar('Loss/avg_rec_loss', avg_rec_loss.item(), global_step)
                
                for mod_idx, loss in enumerate(rec_losses):
                    writer.add_scalar(f'Loss/rec_loss_{modality_names[mod_idx]}', loss.item(), global_step)
                
                log_message = (
                    f"[Dataset: {dataset_config['name']}] [Epoch {epoch}/{opt.epoch}] "
                    f"[Batch {progress[loader_idx]}/{len(train_loaders[loader_idx])}] "
                    f"[Global Batch {i+1}/{total_iterations}] "
                    f"[total_loss: {total_loss.item():.4f}] "
                    f"[nce loss: {nce_loss.item():.4f}] [avg_rec_loss: {avg_rec_loss.item():.4f}] "
                )
                
                for mod_idx, name in enumerate(modality_names):
                    log_message += f"[rec_loss_{name}: {rec_losses[mod_idx].item():.4f}] "
                log_message += f"[lr: {get_lr(optimizer):.6f}]"
                
                print(log_message)
                
                if i % opt.save_output == 0:
                    images_to_save = []
                    for mod_idx, name in enumerate(modality_names):
                        images_to_save.append(modalities[mod_idx][0])
                        images_to_save.append(rec_images[mod_idx][0])
                    
                    save_image(
                        images_to_save,
                        opt.img_save_path + f"{dataset_config['name']}_{epoch}_{i}.png",
                        nrow=len(modality_names),
                        normalize=True,
                    )
                    logging.info(log_message)
        
        if rank == 0:
            print(f"Epoch {epoch} completed. Dataset progress:")
            for idx, d_conf in enumerate(datasets_config):
                print(f"  {d_conf['name']}: {progress[idx]}/{len(train_loaders[idx])} batches")
        
            if epoch % opt.save_weight == 0:
                torch.save(mae.module.state_dict(), opt.weight_save_path + f"{epoch}_MAE.pth")

    if rank == 0:
        torch.save(mae.module.state_dict(), opt.weight_save_path + '/MAE_final.pth')

    cleanup_ddp()

if __name__ == '__main__':
    main()
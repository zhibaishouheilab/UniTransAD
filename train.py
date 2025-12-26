import torch
import logging
import numpy as np
import math
import os
import argparse
from tensorboardX import SummaryWriter
from torchvision.utils import save_image 
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Imports from new structure
from data.dataset import BrainOmniADataset
from models.unitransad import UniTransAD
from configs.options import UniTransADOptions

def setup_ddp():
    dist.init_process_group(backend="nccl") 
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_ddp():
    dist.destroy_process_group()

def adjust_lr_sniping(optimizer, init_lr, epoch, total_epochs, warmup_epochs=2, hold_epochs=8, min_lr=1e-6):
    """
    LR Strategy: Warmup -> Hold (Sniping) -> Decay.
    Helps in stabilizing GAN-like training before convergence.
    """
    if epoch < warmup_epochs:
        lr = init_lr * (epoch + 1) / warmup_epochs
    elif epoch < warmup_epochs + hold_epochs:
        lr = init_lr
    else:
        curr_decay_epoch = epoch - (warmup_epochs + hold_epochs)
        total_decay_epochs = total_epochs - (warmup_epochs + hold_epochs)
        if total_decay_epochs <= 0:
            decay_factor = 0
        else:
            decay_factor = 0.5 * (1 + math.cos(math.pi * curr_decay_epoch / total_decay_epochs))
        lr = min_lr + (init_lr - min_lr) * decay_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    setup_ddp()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    opt = UniTransADOptions().get_opt()

    # NOTE: Update these paths to your actual Brain-OmniA dataset locations
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

    model = UniTransAD(img_size=opt.img_size, patch_size=opt.patch_size, 
                       embed_dim=opt.dim_encoder, depth=opt.depth, 
                       num_heads=opt.num_heads, in_chans=1,
                       decoder_embed_dim=opt.dim_decoder, 
                       decoder_depth=opt.decoder_depth, 
                       decoder_num_heads=opt.decoder_num_heads,
                       mlp_ratio=opt.mlp_ratio, norm_layer=torch.nn.LayerNorm,
                       ema_momentum=opt.ema_momentum)

    if rank == 0:
        os.makedirs(opt.img_save_path, exist_ok=True)
        os.makedirs(opt.weight_save_path, exist_ok=True)

    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    train_loaders = []
    train_samplers = []
    for dataset_config in datasets_config:
        train_dataset = BrainOmniADataset(
            image_root=dataset_config['data_root'],
            edge_root=dataset_config['edge_root'],
            img_size=opt.img_size,
            augment=opt.augment,
            sequences=dataset_config['modality']
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(0.9, 0.95), weight_decay=opt.weight_decay)

    if opt.use_checkpoints:
        if rank == 0: print(f'Rank {rank} loading checkpoint: {opt.checkpoint_path}')
        dist.barrier()
        map_location = {'cuda:0': f'cuda:{rank}'}
        state_dict = torch.load(opt.checkpoint_path, map_location=map_location)
        model.module.load_state_dict(state_dict, strict=False) 

    writer = None
    if rank == 0:
        logging.basicConfig(filename=opt.log_path,
                            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        writer = SummaryWriter(log_dir=opt.logs_path)

    for epoch in range(1, opt.epoch + 1):
        for sampler in train_samplers:
            sampler.set_epoch(epoch)
        
        adjust_lr_sniping(
            optimizer, 
            init_lr=opt.lr, 
            epoch=epoch-1, 
            total_epochs=opt.epoch, 
            warmup_epochs=opt.warmup_epochs, 
            hold_epochs=opt.hold_epochs,
            min_lr=opt.min_lr
        )

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
            
            optimizer.zero_grad()
            outputs = model(modalities, edges, opt.temperature)
            
            # --- Update DSPM (Dynamic Style Prototype Memory) ---
            style_means = outputs['style_means']
            style_stds = outputs['style_stds']
            # Only update if current batch contains relevant sequences
            if set(modality_names).intersection(set(['t1', 't2', 't1ce', 'flair'])):
                model.module.update_dspm(style_means, style_stds, modality_names)

            # --- Losses ---
            loss_cscal = outputs['loss_cscal'] # Was nce_loss
            trans_losses = outputs['trans_losses']
            rec_images = outputs['rec_images']
            
            loss_cscal = loss_cscal * opt.nce_weight
            weighted_trans_losses = [loss * opt.rec_weight for loss in trans_losses]
            loss_trans = sum(weighted_trans_losses) / len(weighted_trans_losses) * 5
            
            total_loss = loss_cscal + loss_trans
            
            total_loss.backward() 
            optimizer.step()      
            
            if rank == 0:
                global_step = (epoch - 1) * total_iterations + i
                writer.add_scalar('Loss/total_loss', total_loss.item(), global_step)
                writer.add_scalar('Loss/cscal_loss', loss_cscal.item(), global_step)
                writer.add_scalar('Loss/trans_loss', loss_trans.item(), global_step)
                
                log_msg = (
                    f"[{dataset_config['name']}] [Epoch {epoch}/{opt.epoch}] "
                    f"[Loss T: {total_loss.item():.4f}] "
                    f"[CSCAL: {loss_cscal.item():.4f}] [Trans: {loss_trans.item():.4f}] "
                    f"[LR: {get_lr(optimizer):.6f}]"
                )
                print(log_msg)
                
                if i % opt.save_output == 0:
                    logging.info(log_msg)
                    # Simple viz logic (saving first sample)
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
        
        if rank == 0:
            print(f"Epoch {epoch} completed.")
            if epoch % opt.save_weight == 0:
                torch.save(model.module.state_dict(), os.path.join(opt.weight_save_path, f"{epoch}_UniTransAD.pth"))

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(opt.weight_save_path, 'UniTransAD_final.pth'))

    cleanup_ddp()

if __name__ == '__main__':
    main()
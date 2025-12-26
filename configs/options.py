import argparse
import os

class UniTransADOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="UniTransAD Configuration")
        
        # 任务名称，用于日志和保存路径
        self.parser.add_argument("--task_name", default='UniTransAD_Run', type=str, 
                               help='Name of the task for logging and saving paths')

        # --- 训练超参数 ---
        # 使用 AdamW，配合梯度累积建议 LR 设置
        self.parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (max)')
        self.parser.add_argument("--batch_size", default=1, type=int, help='batch size per gpu')
        self.parser.add_argument("--accum_iter", default=16, type=int, help='accumulate gradient iterations')
        
        # Epoch 设置 (总计 15 Epoch)
        self.parser.add_argument("--epoch", default=15, type=int, help='number of epochs')
        
        # 调度器策略参数 (Sniping Scheduler: Warmup -> Hold -> Decay)
        self.parser.add_argument("--warmup_epochs", default=2, type=int, help='epochs for linear warmup')
        self.parser.add_argument("--hold_epochs", default=8, type=int, help='epochs to hold max lr')
        self.parser.add_argument("--min_lr", default=1e-6, type=float, help='minimum learning rate')
        
        # 优化器参数
        self.parser.add_argument("--weight_decay", default=0.05, type=float, help='weight decay for AdamW')

        # --- 模型架构参数 (UniTransAD) ---
        self.parser.add_argument("--patch_size", default=8, type=int, help='patch size for embedding')
        self.parser.add_argument("--img_size", default=256, type=int, help='input image size')
        self.parser.add_argument("--depth", default=12, type=int, help='depth of content/style encoders')
        self.parser.add_argument("--decoder_depth", default=8, type=int, help='depth of generator/decoder')
        self.parser.add_argument("--num_heads", default=16, type=int, help='attention heads in encoder')
        self.parser.add_argument("--decoder_num_heads", default=8, type=int, help='attention heads in decoder')
        self.parser.add_argument("--mlp_ratio", default=4, type=int, help='MLP ratio in transformer blocks')
        self.parser.add_argument("--dim_encoder", default=128, type=int, help='dimension of encoder embeddings')
        self.parser.add_argument("--dim_decoder", default=64, type=int, help='dimension of decoder embeddings')
        
        # DSPM 参数 (Dynamic Style Prototype Memory)
        self.parser.add_argument("--ema_momentum", default=0.01, type=float, 
                               help='Momentum for DSPM updates (m in Eq. 4)')

        # --- 数据增强 ---
        self.parser.add_argument("--augment", default=True, type=bool, help='perform data augmentation')
        self.parser.add_argument("--rotation_range", default=3, type=int, help='range of rotations (0-3)')
        self.parser.add_argument("--flip_prob", default=0.5, type=float, help='probability of random flip')
        self.parser.add_argument("--num_workers", default=4, type=int, help='number of data loader workers')

        # --- 损失函数权重 ---
        # nce_weight 对应论文中的 alpha (CSCAL loss weight, Eq. 13)
        self.parser.add_argument("--nce_weight", default=0.005, type=float, help='Weight for L_CSCAL (Contrastive Alignment)')
        self.parser.add_argument("--rec_weight", default=1.0, type=float, help='Weight for L_trans (Reconstruction/Translation)')
        self.parser.add_argument("--temperature", default=0.05, type=float, help='Temperature tau for L_CSCAL (Eq. 6)')

        # --- 日志与保存 ---
        self.parser.add_argument('--use_checkpoints', default=False, type=bool, help='resume training from checkpoints')
        self.parser.add_argument('--checkpoint_path', type=str, default=None, help='path to checkpoint file')
        
        self.parser.add_argument('--img_save_path', type=str, default=None, help='path to save generated images')
        self.parser.add_argument('--weight_save_path', type=str, default=None, help='path to save model weights')
        
        self.parser.add_argument("--save_output", default=200, type=int, help='save output every N batches')
        self.parser.add_argument("--save_weight", default=1, type=int, help='save weights every N epochs')
        
        self.parser.add_argument("--log_path", default=None, help='path to log file')
        self.parser.add_argument("--logs_path", default=None, help='folder to save summary writer logs')

    def get_opt(self):
        self.opt = self.parser.parse_args()
        
        # 动态设置路径
        if self.opt.img_save_path is None:
            self.opt.img_save_path = f'./snapshot/{self.opt.task_name}/'
        if self.opt.weight_save_path is None:
            self.opt.weight_save_path = f'./weight/{self.opt.task_name}/'
        if self.opt.log_path is None:
            self.opt.log_path = f'./log/{self.opt.task_name}.log'
        if self.opt.logs_path is None:
            self.opt.logs_path = f'./log/{self.opt.task_name}'
            
        return self.opt
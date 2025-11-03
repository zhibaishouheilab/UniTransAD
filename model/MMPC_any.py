import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed, Block
from utils.pos_embed import get_2d_sincos_pos_embed
import random
from itertools import combinations

class MultiModalPatchMAE(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128, depth=12, num_heads=8,
                 decoder_embed_dim=64, decoder_depth=6, decoder_num_heads=8, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, style_dim=2):
        super().__init__()
        
        # 用于融合 patch token 与 edge latent 的线性层
        self.edge_fuse = nn.Linear(2 * decoder_embed_dim, decoder_embed_dim)
        
        # Patch embedding（原始图像与边缘图共享）
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_edge = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)
        
        # 风格编码器（所有模态共享）
        style_encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.style_encoder = nn.Sequential(*style_encoder, nn.Linear(embed_dim, 2*decoder_embed_dim))

        # 内容编码器（所有模态共享）
        self.content_encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.sig = nn.Sigmoid()
        self.norm = norm_layer(embed_dim)
        
        self.generator = self.build_generator(decoder_embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, norm_layer)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # 使用 sin-cos 方式初始化位置编码
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        for w_ in [self.patch_embed.proj.weight.data,
                   self.patch_embed_edge.proj.weight.data]:
            torch.nn.init.xavier_uniform_(w_.view([w_.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        
        # 初始化 Linear 和 LayerNorm 层
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def build_generator(self, input_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, norm_layer):
        input_projection = nn.Linear(input_dim, decoder_embed_dim)
        generator_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        generator_pred = nn.Linear(decoder_embed_dim, self.patch_size**2, bias=True)

        return nn.Sequential(input_projection, *generator_blocks, 
                             norm_layer(decoder_embed_dim), generator_pred, self.sig)

    def adain(self, content, style_mean, style_std, eps=1e-6):
        style_mean = style_mean.unsqueeze(1)  # (N, 1, D)
        style_std = style_std.unsqueeze(1)
        content_mean = content.mean(dim=-1, keepdim=True)  # (N, L, 1)
        content_std = content.std(dim=-1, keepdim=True) + eps
        return style_std * (content - content_mean) / content_std + style_mean

    def patchify(self, imgs):
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p 
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    def unpatchify(self, x):
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def compute_style_mean_std(self, embed_M):
        style_features = self.style_encoder(embed_M)
        style_mean, style_std = torch.chunk(style_features, 2, dim=-1)
        style_mean = style_mean.mean(dim=1)
        style_std = style_std.mean(dim=1)
        return style_mean, style_std

    def compute_nce_loss(self, z_i, z_j, mask_i, mask_j):
        B, N, C = z_i.shape
        sim_matrix = F.cosine_similarity(z_i.unsqueeze(2), z_j.unsqueeze(1), dim=-1)

        # 构造前景掩码
        f_mask_i = mask_i.unsqueeze(2)   # (B, N, 1)
        f_mask_j = mask_j.unsqueeze(1)   # (B, 1, N)
        valid_mask_2d = f_mask_i * f_mask_j  # (B, N, N)

        # positives：对角线部分
        positives = torch.diagonal(sim_matrix, dim1=1, dim2=2)
        diag_mask = mask_i & mask_j    
        positives = positives * diag_mask  # (B, N)
        # negatives：非对角线部分
        mask = ~torch.eye(N, dtype=torch.bool, device=z_i.device).unsqueeze(0).expand(B, N, N)
        negatives = sim_matrix[mask].view(B, N, -1)
        negatives = negatives * valid_mask_2d[mask].view(B, N, -1)

        temperature = 0.1
        positives_exp = torch.exp(positives / temperature)
        negatives_exp = torch.exp(negatives / temperature).sum(dim=-1)
        eps = 1e-8
        nce_loss = -torch.log((positives_exp+eps) / (positives_exp + negatives_exp + eps))
        nce_loss = nce_loss.sum() / diag_mask.sum()  # 前景区域平均
        return nce_loss
    
    def fuse_and_generate(self, candidate_modalities, style_mean, style_std, generator, target_edge):
        selected = random.choice(candidate_modalities)
        fused_c = self.decoder_embed(selected)
        
        # 分离 CLS token 和 patch tokens
        cls_token = fused_c[:, :1, :]         # (B, 1, decoder_embed_dim)
        patch_tokens = fused_c[:, 1:, :]        # (B, N, decoder_embed_dim)
        
        # 利用目标模态的边缘图获得边缘 latent
        edge_latent = self.decoder_embed(self.patch_embed_edge(target_edge))

        # 融合 patch tokens 与 edge_latent
        concat_tokens = torch.cat([patch_tokens, edge_latent], dim=-1)
        fused_patch = self.edge_fuse(concat_tokens)

        # 组合 CLS token 与融合后的 patch tokens
        fused_c = torch.cat([cls_token, fused_patch], dim=1)

        # 加入位置编码
        fused_c = fused_c + self.decoder_pos_embed[:, :fused_c.size(1), :]
        # 自适应实例归一化
        fused = self.adain(fused_c, style_mean, style_std)
        # 生成 patch 重构，去掉 CLS token 后返回
        p_gen = generator(fused)
        p_gen = p_gen[:, 1:, :]
        return p_gen

    def forward(self, modalities, edges):
        """
        支持任意数量（>=2）的模态训练
        modalities: 模态图像列表 [M1, M2, ...] 每个元素形状为 (B, 1, H, W)
        edges: 对应边缘图列表 [M1_edge, M2_edge, ...] 每个元素形状为 (B, 1, H, W)
        """
        num_mods = len(modalities)
        assert num_mods >= 2, "至少需要2个模态"
        assert len(edges) == num_mods, "模态和边缘图数量必须匹配"
        
        # 1. 预处理所有模态
        patches = []        # 存储每个模态的patch表示
        valid_masks = []    # 存储每个模态的前景掩码
        embeddings = []     # 存储每个模态的嵌入
        style_means = []    # 存储每个模态的风格均值
        style_stds = []     # 存储每个模态的风格标准差
        
        # 处理每个模态
        for i, mod in enumerate(modalities):
            # 1.1 对原始图像做patchify
            p = self.patchify(mod)
            patches.append(p)
            
            # 1.2 计算前景掩码（基于原始图像patch）
            valid_mask = (p.var(dim=-1) > 0)  # (B, L)
            valid_masks.append(valid_mask)
            
            # 1.3 做patch_embed + CLS + pos
            embed = self.patch_embed(mod)
            embed = torch.cat([self.cls_token.expand(embed.shape[0], -1, -1), embed], dim=1)
            embed = embed + self.pos_embed
            embeddings.append(embed)
            
            # 1.4 计算风格信息
            style_mean, style_std = self.compute_style_mean_std(embed)
            style_means.append(style_mean)
            style_stds.append(style_std)
        
        # 2. 内容编码：所有模态统一使用同一内容编码器
        content_embeddings = []
        for embed in embeddings:
            z = embed
            for blk in self.content_encoder:
                z = blk(z)
            content_embeddings.append(z)
        
        nce_losses = []
        
        # 遍历所有模态对组合
        for i, j in combinations(range(num_mods), 2):
            # 获取内容嵌入（去掉CLS token）
            z_i = content_embeddings[i][:, 1:, :]
            z_j = content_embeddings[j][:, 1:, :]
            
            # NCE损失
            nce_ij = self.compute_nce_loss(z_i, z_j, valid_masks[i], valid_masks[j])
            nce_losses.append(nce_ij)
        
        nce_loss = sum(nce_losses) / len(nce_losses) if nce_losses else torch.tensor(0.0)
        
        # 4. 针对每个目标模态生成重构
        rec_losses = []
        rec_patches = []
        rec_images = []
        
        for i in range(num_mods):
            # 使用所有内容嵌入作为候选，目标模态的风格和边缘图
            p_gen = self.fuse_and_generate(
                content_embeddings, 
                style_means[i], 
                style_stds[i], 
                self.generator, 
                edges[i]
            )
            
            # 计算重构损失
            rec_loss = F.l1_loss(p_gen, patches[i])
            rec_losses.append(rec_loss)
            rec_patches.append(p_gen)
            
            # 解码patch重构结果，恢复为完整图像
            rec_img = self.unpatchify(p_gen)
            rec_images.append(rec_img)
        
        rec_loss_total = sum(rec_losses)
        
        # 5. 返回结果
        results = {
            'nce_loss': nce_loss,
            'rec_loss_total': rec_loss_total,
            'rec_losses': rec_losses,
            'rec_images': rec_images
        }
        
        # 添加每个模态的详细输出
        for i in range(num_mods):
            results[f'rec_loss_M{i+1}'] = rec_losses[i]
            results[f'M{i+1}_gen'] = rec_images[i]
        
        return results
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed, Block
from models.pos_embed import get_2d_sincos_pos_embed
import random
from itertools import combinations

class UniTransAD(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=128, depth=12, num_heads=8,
                 decoder_embed_dim=64, decoder_depth=6, decoder_num_heads=8, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, style_dim=2, 
                 ema_momentum=0.1): 
        """
        UniTransAD: Universal Translation Framework for Anomaly Detection in Brain MRI.
        Includes Content-Style Disentanglement and Dynamic Style Prototype Memory (DSPM).
        """
        super().__init__()
        
        # Edge Fusion Layer
        self.edge_fuse = nn.Linear(2 * decoder_embed_dim, decoder_embed_dim)
        
        # Patch Embeddings (Shared between Image and Edge)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.patch_embed_edge = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)
        
        # Style Encoder (Shared E_s)
        style_encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.style_encoder = nn.Sequential(*style_encoder, nn.Linear(embed_dim, 2*decoder_embed_dim))

        # Content Encoder (Shared E_c)
        self.content_encoder = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        
        # Decoder Projection & Pos Embed
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)
        
        self.sig = nn.Sigmoid()
        self.norm = norm_layer(embed_dim)
        
        # Generator (G)
        self.generator = self.build_generator(decoder_embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, norm_layer)

        # --- Dynamic Style Prototype Memory (DSPM) ---
        # Stores global style prototypes for sequences
        self.ema_momentum = ema_momentum
        self.dspm_sequences = ['t1', 't2', 't1ce', 'flair']
        for mod_name in self.dspm_sequences:
            self.register_buffer(f'dspm_{mod_name}_mean', torch.zeros(decoder_embed_dim))
            self.register_buffer(f'dspm_{mod_name}_std', torch.zeros(decoder_embed_dim))

        self.initialize_weights()

    def initialize_weights(self):
        # Use sin-cos position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**0.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        for w_ in [self.patch_embed.proj.weight.data,
                   self.patch_embed_edge.proj.weight.data]:
            torch.nn.init.xavier_uniform_(w_.view([w_.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def update_dspm(self, current_means, current_stds, seq_names):
        """
        Update the Dynamic Style Prototype Memory via EMA.
        Called during training (Eq. 4 & 5).
        """
        if not self.training:
            return
            
        momentum = self.ema_momentum
        
        for i, mod_name in enumerate(seq_names):
            if mod_name in self.dspm_sequences:
                mean_buffer = getattr(self, f'dspm_{mod_name}_mean')
                std_buffer = getattr(self, f'dspm_{mod_name}_std')
                
                # Average across batch
                batch_mean_avg = current_means[i].mean(dim=0)
                batch_std_avg = current_stds[i].mean(dim=0)

                # EMA Update
                new_mean = (momentum * batch_mean_avg) + ((1 - momentum) * mean_buffer)
                new_std = (momentum * batch_std_avg) + ((1 - momentum) * std_buffer)
                
                mean_buffer.copy_(new_mean)
                std_buffer.copy_(new_std)
            
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
        """Adaptive Instance Normalization."""
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

    def compute_cscal_loss(self, z_i, z_j, mask_i, mask_j, temperature):
        """
        Cross-Sequence Contrastive Alignment Loss (L_CSCAL).
        Eq. 6 in the paper.
        """
        B, N, C = z_i.shape
        sim_matrix = F.cosine_similarity(z_i.unsqueeze(2), z_j.unsqueeze(1), dim=-1)

        f_mask_i = mask_i.unsqueeze(2)
        f_mask_j = mask_j.unsqueeze(1)
        valid_mask_2d = f_mask_i * f_mask_j

        # Positives (Same spatial location)
        positives = torch.diagonal(sim_matrix, dim1=1, dim2=2)
        diag_mask = mask_i & mask_j    
        positives = positives * diag_mask

        # Negatives (Different spatial locations)
        mask = ~torch.eye(N, dtype=torch.bool, device=z_i.device).unsqueeze(0).expand(B, N, N)
        negatives = sim_matrix[mask].view(B, N, -1)
        negatives = negatives * valid_mask_2d[mask].view(B, N, -1)

        positives_exp = torch.exp(positives / temperature)
        negatives_exp = torch.exp(negatives / temperature).sum(dim=-1)
        eps = 1e-8
        loss = -torch.log((positives_exp+eps) / (positives_exp + negatives_exp + eps))
        loss = loss.sum() / diag_mask.sum()
        return loss
    
    def fuse_and_generate(self, candidate_modalities, style_mean, style_std, generator, target_edge):
        """
        Fuse content, edge and style to generate translation.
        Eq. 10 & 11 in the paper.
        """
        selected = random.choice(candidate_modalities)
        fused_c = self.decoder_embed(selected)
        
        cls_token = fused_c[:, :1, :]
        patch_tokens = fused_c[:, 1:, :]
        
        # Edge Guidance (Eq. 9)
        edge_latent = self.decoder_embed(self.patch_embed_edge(target_edge))
        concat_tokens = torch.cat([patch_tokens, edge_latent], dim=-1)
        fused_patch = self.edge_fuse(concat_tokens)

        fused_c = torch.cat([cls_token, fused_patch], dim=1)
        fused_c = fused_c + self.decoder_pos_embed[:, :fused_c.size(1), :]
        
        # AdaIN Injection (Eq. 11)
        fused = self.adain(fused_c, style_mean, style_std)
        
        p_gen = generator(fused)
        p_gen = p_gen[:, 1:, :] # Remove CLS token
        return p_gen

    def forward(self, modalities, edges, temperature):
        """
        Args:
            modalities: List of tensors [M1, M2, ...]
            edges: List of edge tensors [E1, E2, ...]
            temperature: scalar for CSCAL loss
        """
        num_mods = len(modalities)
        assert num_mods >= 2, "Requires at least 2 modalities for training translation."
        
        patches, valid_masks, embeddings = [], [], []
        style_means, style_stds = [], []
        
        # 1. Encoding (Shared E_s, E_c)
        for i, mod in enumerate(modalities):
            p = self.patchify(mod)
            patches.append(p)
            valid_masks.append((p.var(dim=-1) > 0))
            
            embed = self.patch_embed(mod)
            embed = torch.cat([self.cls_token.expand(embed.shape[0], -1, -1), embed], dim=1)
            embed = embed + self.pos_embed
            embeddings.append(embed)
            
            s_mean, s_std = self.compute_style_mean_std(embed)
            style_means.append(s_mean)
            style_stds.append(s_std)
        
        content_embeddings = []
        for embed in embeddings:
            z = embed
            for blk in self.content_encoder: z = blk(z)
            content_embeddings.append(z)
        
        # 2. CSCAL Loss (L_CSCAL)
        cscal_losses = []
        for i, j in combinations(range(num_mods), 2):
            z_i = content_embeddings[i][:, 1:, :]
            z_j = content_embeddings[j][:, 1:, :]
            cscal_losses.append(self.compute_cscal_loss(z_i, z_j, valid_masks[i], valid_masks[j], temperature))
        
        loss_cscal = sum(cscal_losses) / len(cscal_losses) if cscal_losses else torch.tensor(0.0)
        
        # 3. Translation Generation & Loss (L_trans)
        trans_losses = []
        rec_images = []
        
        for i in range(num_mods):
            # Target: Modality i. Source: Randomly selected from others (handled inside)
            p_gen = self.fuse_and_generate(
                content_embeddings, 
                style_means[i], 
                style_stds[i], 
                self.generator, 
                edges[i]
            )
            
            trans_losses.append(F.l1_loss(p_gen, patches[i]))
            rec_images.append(self.unpatchify(p_gen))
        
        loss_trans_total = sum(trans_losses)
        
        results = {
            'loss_cscal': loss_cscal,        # Renamed from nce_loss
            'loss_trans': loss_trans_total,  # Renamed from rec_loss_total
            'trans_losses': trans_losses,    # Individual losses
            'rec_images': rec_images,
            'style_means': style_means,      # For DSPM update
            'style_stds': style_stds
        }
        
        return results
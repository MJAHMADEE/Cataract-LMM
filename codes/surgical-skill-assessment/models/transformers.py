import torch
import torch.nn as nn
from einops import rearrange


class TransformerBlock(nn.Module):
    """Basic transformer block with self-attention and MLP."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class MViT(nn.Module):
    """Multiscale Vision Transformer for video understanding."""

    def __init__(
        self,
        num_classes,
        img_size=224,
        patch_size=16,
        num_frames=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding with temporal dimension
        self.patch_embed = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames * self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Transformer blocks with multi-scale pooling
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        # Pooling layers for multi-scale features
        self.pool_layers = nn.ModuleList(
            [
                nn.MaxPool1d(kernel_size=2, stride=2) if i % 3 == 0 else nn.Identity()
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T, H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed[:, : x.size(1)]

        # Apply transformer blocks with multi-scale pooling
        for i, (block, pool) in enumerate(zip(self.blocks, self.pool_layers)):
            x = block(x)
            if not isinstance(pool, nn.Identity) and x.size(1) > 1:
                # Apply pooling to all tokens except cls token
                cls_token, tokens = x[:, :1], x[:, 1:]
                tokens = rearrange(tokens, "b n d -> b d n")
                tokens = pool(tokens)
                tokens = rearrange(tokens, "b d n -> b n d")
                x = torch.cat([cls_token, tokens], dim=1)

        x = self.norm(x)

        # Use cls token for classification
        x = x[:, 0]
        x = self.head(x)

        return x


class VideoMAE(nn.Module):
    """Video Masked Autoencoder V2 - adapted for classification."""

    def __init__(
        self,
        num_classes,
        img_size=224,
        patch_size=16,
        num_frames=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames

        # Patch embedding (3D for video)
        self.patch_embed = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size),
        )

        num_patches = (num_frames // 2) * (img_size // patch_size) ** 2
        self.num_patches = num_patches

        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed[:, : x.size(1)]

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use cls token for classification
        x = x[:, 0]
        x = self.head(x)

        return x


class FactorizedTransformerBlock(nn.Module):
    """Transformer block with factorized space-time attention."""

    def __init__(
        self,
        dim,
        num_heads,
        num_time_patches,
        num_space_patches,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_time_patches = num_time_patches
        self.num_space_patches = num_space_patches

        # Spatial attention
        self.norm1 = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Temporal attention
        self.norm2 = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # MLP
        self.norm3 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, T, HW):
        B, N, D = x.shape

        # Separate cls token
        cls_token, x_patches = x[:, :1], x[:, 1:]

        # Spatial attention (within each frame)
        x_spatial = rearrange(x_patches, "b (t hw) d -> (b t) hw d", t=T, hw=HW)
        x_spatial = self.norm1(x_spatial)
        x_spatial, _ = self.spatial_attn(x_spatial, x_spatial, x_spatial)
        x_patches = x_patches + rearrange(
            x_spatial, "(b t) hw d -> b (t hw) d", b=B, t=T
        )

        # Temporal attention (across frames)
        x_temporal = rearrange(x_patches, "b (t hw) d -> (b hw) t d", t=T, hw=HW)
        x_temporal = self.norm2(x_temporal)
        x_temporal, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_patches = x_patches + rearrange(
            x_temporal, "(b hw) t d -> b (t hw) d", b=B, hw=HW
        )

        # Recombine with cls token
        x = torch.cat([cls_token, x_patches], dim=1)

        # MLP
        x = x + self.mlp(self.norm3(x))

        return x


class ViViT(nn.Module):
    """Video Vision Transformer with factorized self-attention."""

    def __init__(
        self,
        num_classes,
        img_size=224,
        patch_size=16,
        num_frames=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Tubelet embedding (3D patches)
        self.patch_embed = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(2, patch_size, patch_size),
            stride=(2, patch_size, patch_size),
        )

        # Calculate number of spatiotemporal patches
        self.num_time_patches = num_frames // 2
        self.num_space_patches = self.num_patches

        # Positional embeddings (separate for space and time)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.space_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_space_patches, embed_dim)
        )
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_time_patches, embed_dim)
        )

        # Transformer blocks with factorized attention
        self.blocks = nn.ModuleList(
            [
                FactorizedTransformerBlock(
                    embed_dim,
                    num_heads,
                    self.num_time_patches,
                    self.num_space_patches,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.space_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        B = x.shape[0]

        # Tubelet embedding
        x = self.patch_embed(x)  # (B, embed_dim, T', H', W')
        T_new, H_new, W_new = x.shape[2:]
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # Add spatial and temporal position embeddings
        space_pos = self.space_pos_embed.repeat(
            1, T_new, 1
        )  # Repeat for each time step
        time_pos = self.time_pos_embed.repeat_interleave(
            H_new * W_new, dim=1
        )  # Repeat for each spatial position
        x = x + space_pos[:, : x.size(1)] + time_pos[:, : x.size(1)]

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, T_new, H_new * W_new)

        x = self.norm(x)

        # Use cls token for classification
        x = x[:, 0]
        x = self.head(x)

        return x

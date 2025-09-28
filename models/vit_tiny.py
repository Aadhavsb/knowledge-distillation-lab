import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """Patch Embedding layer for Vision Transformer."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism."""

    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """MLP block for Transformer."""

    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.1):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and MLP."""

    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) implementation."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1):
        super(VisionTransformer, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.n_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        # Initialize positional embeddings
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize other weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, n_patches + 1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Use class token for classification
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)

        return x

    def get_attention_layers(self):
        """Get all attention layers for pruning."""
        attn_layers = []
        for block in self.blocks:
            attn_layers.append(block.attn.qkv)
            attn_layers.append(block.attn.proj)
        return attn_layers

    def get_mlp_layers(self):
        """Get all MLP layers for pruning."""
        mlp_layers = []
        for block in self.blocks:
            mlp_layers.append(block.mlp.fc1)
            mlp_layers.append(block.mlp.fc2)
        mlp_layers.append(self.head)
        return mlp_layers


def get_vit_tiny_cifar10(pretrained=False, num_classes=10):
    """
    Get ViT-Tiny model for CIFAR-10.

    Args:
        pretrained (bool): If True, try to load pre-trained weights
        num_classes (int): Number of output classes

    Returns:
        ViT-Tiny model
    """
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1
    )

    if pretrained:
        try:
            # Try to load pre-trained weights using timm
            import timm
            pretrained_model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

            # Copy compatible weights (this is a simplified approach)
            # In practice, you'd need careful weight adaptation
            print("Warning: Using from-scratch initialization as pre-trained weight adaptation is complex")

        except ImportError:
            print("timm not available, using from-scratch initialization")
        except Exception as e:
            print(f"Could not load pre-trained weights: {e}")
            print("Using from-scratch initialization")

    return model


def get_vit_tiny_pretrained_timm(num_classes=10):
    """
    Get ViT-Tiny model using timm library with pre-trained weights.
    This requires timm to be installed: pip install timm
    """
    try:
        import timm

        # Load pre-trained ViT-Tiny
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

        # Adapt for CIFAR-10
        # Note: This is a simplified adaptation - in practice you might need
        # to adjust patch embedding and positional embeddings for 32x32 images
        model.head = nn.Linear(model.head.in_features, num_classes)

        return model

    except ImportError:
        raise ImportError("timm library is required for pre-trained ViT. Install with: pip install timm")



"""
Implementation of the Vision Transformer (ViT) model.

This module implements the Vision Transformer architecture as described in the paper
'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale'
(https://arxiv.org/abs/2010.11929).

The implementation includes:
- Patch Embedding
- Position Embedding
- Multi-Head Self-Attention
- Transformer Encoder
- Classification Head
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class PatchEmbedding(nn.Module):
    """
    Converts input images into patch embeddings.
    
    This module:
    1. Splits the input image into fixed-size patches
    2. Projects these patches into an embedding space
    3. Adds a learnable classification token
    4. Adds learnable position embeddings
    
    Args:
        image_size (int): Size of the input image (assumed square)
        patch_size (int): Size of each patch (assumed square)
        in_channels (int): Number of input channels
        embed_dim (int): Dimension of the patch embeddings
    """
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, embed_dim)
        )
        
        # Learnable classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))

    def forward(self, x):
        """
        Forward pass of the PatchEmbedding module.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Patch embeddings with position encoding and cls token
                         Shape: (batch_size, num_patches + 1, embed_dim)
        """
        b = x.shape[0]  # batch size
        x = self.projection(x)
        
        # Add classification token to each sequence
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head Self Attention module.
    
    Implements the multi-head self-attention mechanism from the transformer architecture.
    
    Args:
        embed_dim (int): Dimension of input and output embeddings
        num_heads (int): Number of attention heads
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Combined projection for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        Forward pass of Multi-head Self Attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim)
            
        Returns:
            torch.Tensor: Attention output of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, num_patches, embed_dim = x.shape
        
        # Project input into Q, K, V vectors
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_patches, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        dk = self.head_dim ** 0.5  # Scaling factor
        attention = (q @ k.transpose(-2, -1)) / dk
        attention = attention.softmax(dim=-1)
        attention = self.att_drop(attention)
        
        # Combine attention scores with values and project
        x = (attention @ v).transpose(1, 2).reshape(batch_size, num_patches, embed_dim)
        x = self.projection(x)
        return x

class MLP(nn.Module):
    """
    Multilayer Perceptron module.
    
    A simple feed-forward network applied after attention.
    
    Args:
        in_features (int): Number of input features
        hidden_features (int): Number of hidden features
        out_features (int): Number of output features
        dropout (float): Dropout probability
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Processed tensor
        """
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block module.
    
    Combines multi-head self attention with a feed forward network.
    Uses layer normalization and residual connections.
    
    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            in_features=embed_dim,
            hidden_features=int(embed_dim * mlp_ratio),
            out_features=embed_dim,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Processed tensor
        """
        # Attention branch with residual connection
        x = x + self.attn(self.norm1(x))
        # MLP branch with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) model.
    
    Implements the complete Vision Transformer architecture for image classification.
    
    Args:
        image_size (int): Input image size (assumed square)
        patch_size (int): Patch size (assumed square)
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for classification
        embed_dim (int): Embedding dimension
        depth (int): Number of transformer blocks
        num_heads (int): Number of attention heads
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        dropout (float): Dropout probability
    """
    def __init__(
        self, 
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        # Transformer Encoder
        self.transformer = nn.Sequential(*[
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        # Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass of the Vision Transformer.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Classification logits of shape (batch_size, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        x = self.transformer(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use [CLS] token only
        x = self.head(x)
        
        return x

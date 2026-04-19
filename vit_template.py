"""
CS 676 – Deep Learning  |  Spring 2026
Assignment 1: Vision Transformers
==================================

Instructions
------------
1.  Set STUDENT_ID below to your numeric university ID.  Do this FIRST.
    Every random choice in this assignment (data subset, weight init,
    training trajectory) is derived from this value.  The autograder
    re-derives it from your submission; mismatches lose points.

2.  Implement the TODO sections:
      - Part 1 (TODOs 1.1--1.6): architecture and training pipeline.
      - Part 3 (TODOs 3.1--3.4): quantitative analysis functions.
    Do NOT modify anything outside the TODO blocks.

3.  Do NOT rename classes or functions, and do NOT change any signature
    (argument names, order, or return type contract).  The autograder
    imports this file and calls functions by name.

4.  Once Part 1 is complete, run:
        python vit_template.py --mode all

    This trains the baseline model, runs all ablation experiments, and
    computes all analysis outputs.  Inspect the generated JSON files in
    training_log.json, ablation_logs/, and analysis/ to write your report.

5.  Implement Part 3 (TODOs 3.1--3.4) after Part 1.  Re-run with
        python vit_template.py --mode all
    to regenerate the analysis/ outputs before submitting.

6.  Validate all JSON output before submitting:
        python -m json.tool training_log.json
"""

# =============================================================================
# >>>  SET YOUR STUDENT ID HERE  <<<
# =============================================================================
STUDENT_ID = 50652406          # TODO: replace 0 with your numeric student ID
# =============================================================================

import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Seed utilities  (do not modify)
# ---------------------------------------------------------------------------

def get_seed() -> int:
    """
    Derive a deterministic integer seed from STUDENT_ID.

    The transformation is intentionally simple so you can reproduce it
    by hand, but non-trivial enough that different IDs give well-separated
    seeds.

    Returns
    -------
    int
        A non-negative integer seed in [0, 2**31).
    """
    if STUDENT_ID == 0:
        raise ValueError(
            "STUDENT_ID is still 0.  Open vit_template.py and set it to "
            "your numeric university ID before running anything."
        )
    # Mix the bits so that consecutive IDs don't share a common prefix.
    x = int(STUDENT_ID)
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = ((x >> 16) ^ x) * 0x45D9F3B
    x = (x >> 16) ^ x
    return abs(x) % (2 ** 31)


def set_all_seeds(seed: int) -> None:
    """
    Seed Python's random, NumPy, and PyTorch (CPU) in one call.

    Call this at the start of any function that uses randomness so that
    results are fully reproducible given the same seed.

    Parameters
    ----------
    seed : int
        The seed value returned by get_seed() (or any integer).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Baseline hyper-parameters  (do not modify the dict itself)
# ---------------------------------------------------------------------------

BASELINE_CONFIG: Dict = {
    "patch_size":  4,
    "embed_dim":   64,
    "num_heads":   2,
    "num_layers":  4,
    "mlp_dim":     128,
    "dropout":     0.0,
    "lr":          1e-3,
    "batch_size":  64,
    "epochs":      20,
}

# Epochs at which to save checkpoints during baseline training.
CHECKPOINT_EPOCHS: Tuple[int, ...] = (5, 10, 20)

# CIFAR-10 class names for human-readable output (index == class label).
CIFAR10_CLASSES: Tuple[str, ...] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# =============================================================================
# PART 1 – Model Implementation
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Split an image into non-overlapping patches and project each patch
    into a D-dimensional embedding vector.

    The projection is implemented as a single 2-D convolution whose kernel
    size equals the patch size and whose stride also equals the patch size.
    This is mathematically equivalent to flattening each patch and applying
    a linear layer, but is cleaner and slightly faster in PyTorch.

    Parameters
    ----------
    img_size   : int   – Spatial size of the (square) input image.  Default 32.
    patch_size : int   – Size of each (square) patch in pixels.
    in_chans   : int   – Number of input channels (3 for RGB).
    embed_dim  : int   – Output embedding dimension D.

    Attributes
    ----------
    num_patches : int  – Number of patches N = (img_size // patch_size) ** 2.
    proj        : nn.Conv2d – The patch projection layer.

    Forward
    -------
    Input  : x  –  Tensor of shape (B, C, H, W)
    Output :        Tensor of shape (B, N, D)
                    where N = (img_size // patch_size) ** 2

    Shape walkthrough (example: img_size=32, patch_size=4, D=64)
    -------------------------------------------------------------
    Input       (B, 3, 32, 32)
    After conv  (B, 64, 8, 8)   # 8 = 32 // 4 patches per side
    After flat  (B, 64, 64)     # flatten spatial dims: 8*8 = 64 patches
    After trans (B, 64, 64)     # transpose to (B, N, D)
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # TODO 1.1 ── Define self.proj as an nn.Conv2d that:
        #   • accepts in_chans input channels
        #   • outputs embed_dim channels
        #   • uses kernel_size = patch_size
        #   • uses stride     = patch_size
        #   • has no padding (padding=0)
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1.1 ── Apply self.proj to x, then reshape the output from
        #   (B, D, G, G) → (B, N, D) where G = img_size // patch_size
        #   and N = G * G.
        #
        #   Hint: after the conv you have shape (B, D, G, G).
        #   Call .flatten(2) to get (B, D, N), then .transpose(1, 2) for (B, N, D).
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x


# ---------------------------------------------------------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (MHSA) implemented from scratch.

    Do NOT use nn.MultiheadAttention.  You must write the Q/K/V projections,
    the scaled dot-product, and the output projection yourself.

    Parameters
    ----------
    embed_dim : int   – Token embedding dimension D.  Must be divisible by num_heads.
    num_heads : int   – Number of attention heads h.
    dropout   : float – Dropout probability applied to attention weights.

    Attributes to create
    --------------------
    head_dim  : int   – Dimension per head: D // h.
    scale     : float – Scaling factor 1 / sqrt(head_dim).
    q_proj    : nn.Linear(D, D, bias=True)
    k_proj    : nn.Linear(D, D, bias=True)
    v_proj    : nn.Linear(D, D, bias=True)
    out_proj  : nn.Linear(D, D, bias=True)
    attn_drop : nn.Dropout

    Forward
    -------
    Input  : x          – Tensor of shape (B, T, D)  where T = N + 1 (includes CLS)
    Output : (out, attn_weights)
        out          – Tensor of shape (B, T, D)
        attn_weights – Tensor of shape (B, h, T, T)  ← post-softmax weights

    Implementation guide
    --------------------
    1.  Project x to Q, K, V each of shape (B, T, D).
    2.  Reshape each to (B, T, h, head_dim) then transpose to (B, h, T, head_dim).
    3.  Compute raw scores: scores = Q @ K.T * scale  → shape (B, h, T, T)
    4.  Apply softmax over the last dimension → attn_weights  (B, h, T, T)
    5.  Apply dropout to attn_weights.
    6.  Weighted sum: context = attn_weights @ V  → (B, h, T, head_dim)
    7.  Reshape context back to (B, T, D) (concatenate heads).
    8.  Apply out_proj.
    9.  Return (out, attn_weights).

    Verification
    ------------
    attn_weights should sum to 1.0 along dim=-1 for every (b, h, t).
    Check with: assert attn_weights.sum(-1).allclose(torch.ones(...))
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        # TODO 1.2 ── Create the four linear layers and the dropout layer
        #   described in the docstring above.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO 1.2 ── Implement MHSA following the 9-step guide in the docstring.
        #
        #   Useful ops:
        #     tensor.reshape(B, T, self.num_heads, self.head_dim)
        #     tensor.transpose(1, 2)          # swap T and h dimensions
        #     torch.matmul  or  the @ operator
        #     F.softmax(scores, dim=-1)
        #     tensor.transpose(1, 2).contiguous().reshape(B, T, self.embed_dim)
        B, T, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = q.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().reshape(B, T, self.embed_dim)
        
        out = self.out_proj(context)
        return out, attn_weights

# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    One pre-norm Transformer encoder block.

    The block applies two sub-layers in sequence, each wrapped with a
    pre-LayerNorm and a residual connection:

        x  →  x + MHSA( LN(x) )          ← attention sub-layer
           →  x + MLP(  LN(x) )           ← feed-forward sub-layer

    The MLP is a two-layer network:
        Linear(D → mlp_dim) → GELU → Dropout → Linear(mlp_dim → D) → Dropout

    Parameters
    ----------
    embed_dim : int   – Token embedding dimension D.
    num_heads : int   – Number of attention heads passed to MHSA.
    mlp_dim   : int   – Hidden dimension of the MLP (typically 2× or 4× D).
    dropout   : float – Dropout probability used inside MHSA and the MLP.

    Attributes to create
    --------------------
    norm1  : nn.LayerNorm(embed_dim)
    attn   : MultiHeadSelfAttention(embed_dim, num_heads, dropout)
    norm2  : nn.LayerNorm(embed_dim)
    mlp    : nn.Sequential  (see MLP structure above)

    Forward
    -------
    Input  : x  – Tensor of shape (B, T, D)
    Output : (x_out, attn_weights)
        x_out        – Tensor of shape (B, T, D)
        attn_weights – Tensor of shape (B, h, T, T)  forwarded from MHSA
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim:   int,
        dropout:   float = 0.0,
    ) -> None:
        super().__init__()

        # TODO 1.3 ── Create norm1, attn, norm2, and mlp.
        #
        #   mlp should be an nn.Sequential with exactly these layers in order:
        #     nn.Linear(embed_dim, mlp_dim)
        #     nn.GELU()
        #     nn.Dropout(dropout)
        #     nn.Linear(mlp_dim, embed_dim)
        #     nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO 1.3 ── Implement the pre-norm residual block.
        #
        #   Step 1 (attention sub-layer):
        #     normed      = self.norm1(x)
        #     attn_out, attn_weights = self.attn(normed)
        #     x           = x + attn_out
        #
        #   Step 2 (MLP sub-layer):
        #     x           = x + self.mlp(self.norm2(x))
        #
        #   Return (x, attn_weights).
        normed = self.norm1(x)
         
        attn_out, attn_weights = self.attn(normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """
    Full Vision Transformer (ViT) for image classification.

    Architecture
    ------------
    1. PatchEmbedding converts the image to N patch tokens of dim D.
    2. A learnable CLS token is prepended → sequence length T = N + 1.
    3. Learnable 1-D positional embeddings (shape 1 × T × D) are added.
    4. L TransformerBlocks process the sequence.
    5. A final LayerNorm is applied.
    6. The CLS token (index 0) is extracted and passed through a linear
       classification head to produce class logits.

    Parameters
    ----------
    img_size    : int   – Spatial size of the (square) input image.
    patch_size  : int   – Patch size P.
    in_chans    : int   – Number of input channels.
    num_classes : int   – Number of output classes.
    embed_dim   : int   – Token embedding dimension D.
    num_heads   : int   – Number of attention heads per block.
    num_layers  : int   – Number of Transformer blocks L.
    mlp_dim     : int   – MLP hidden dimension inside each block.
    dropout     : float – Dropout probability.

    Attributes to create
    --------------------
    patch_embed  : PatchEmbedding
    cls_token    : nn.Parameter  shape (1, 1, D)  – initialise with zeros
    pos_embed    : nn.Parameter  shape (1, N+1, D) – initialise with zeros
                   (in practice random init also works; zeros is simpler to test)
    blocks       : nn.ModuleList of L TransformerBlocks
    norm         : nn.LayerNorm(embed_dim)
    head         : nn.Linear(embed_dim, num_classes)

    Forward
    -------
    Input  : x  – Tensor of shape (B, C, H, W)
    Output : (logits, attn_list)
        logits    – Tensor of shape (B, num_classes)
        attn_list – Python list of L tensors, each of shape (B, h, T, T)
                    containing the attention weights from each TransformerBlock,
                    in order from block 0 to block L-1.

    Implementation steps
    --------------------
    1.  x = patch_embed(x)                  → (B, N, D)
    2.  Expand cls_token to (B, 1, D) using .expand(B, -1, -1)
    3.  x = torch.cat([cls_tokens, x], dim=1)  → (B, N+1, D)
    4.  x = x + pos_embed                   → (B, N+1, D)
    5.  For each block in self.blocks:
            x, attn = block(x)
            append attn to attn_list
    6.  x = self.norm(x)
    7.  cls_out = x[:, 0]                   → (B, D)   # CLS token
    8.  logits  = self.head(cls_out)        → (B, num_classes)
    9.  return (logits, attn_list)
    """

    def __init__(
        self,
        img_size:    int   = 32,
        patch_size:  int   = 4,
        in_chans:    int   = 3,
        num_classes: int   = 10,
        embed_dim:   int   = 64,
        num_heads:   int   = 2,
        num_layers:  int   = 4,
        mlp_dim:     int   = 128,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        # Store config so we can reconstruct the model from a checkpoint.
        self.config = {
            "patch_size":  patch_size,
            "embed_dim":   embed_dim,
            "num_heads":   num_heads,
            "num_layers":  num_layers,
            "mlp_dim":     mlp_dim,
            "dropout":     dropout,
        }

        # TODO 1.4 ── Create patch_embed, cls_token, pos_embed, blocks, norm, head.
        #
        #   num_patches = patch_embed.num_patches after you create patch_embed.
        #   pos_embed shape must be (1, num_patches + 1, embed_dim).
        #
        #   Initialise cls_token and pos_embed with torch.zeros (wrapped in
        #   nn.Parameter so they are learnable).
        #
        #   blocks is an nn.ModuleList; create num_layers TransformerBlock instances.
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # TODO 1.4 ── Follow the 9-step guide in the docstring.
        x = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        attn_list: List[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_list.append(attn)
        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits, attn_list

# =============================================================================
# Helper: build a VisionTransformer from a config dict
# =============================================================================

def build_model(config: Dict, num_classes: int = 10) -> VisionTransformer:
    """
    Convenience factory that constructs a VisionTransformer from a flat
    config dictionary (as stored in checkpoints and log files).

    Parameters
    ----------
    config      : dict  – Must contain keys: patch_size, embed_dim, num_heads,
                          num_layers, mlp_dim, dropout.
    num_classes : int   – Classification head width.

    Returns
    -------
    VisionTransformer (on CPU, in eval mode).
    """
    return VisionTransformer(
        img_size    = 32,
        patch_size  = config["patch_size"],
        in_chans    = 3,
        num_classes = num_classes,
        embed_dim   = config["embed_dim"],
        num_heads   = config["num_heads"],
        num_layers  = config["num_layers"],
        mlp_dim     = config["mlp_dim"],
        dropout     = config["dropout"],
    )


# =============================================================================
# PART 1 – Data and Training
# =============================================================================

def get_cifar10_subset(
    data_root: str = "./data",
) -> Tuple[Subset, datasets.CIFAR10]:
    """
    Load CIFAR-10 and return a class-balanced training subset plus the full
    test set.

    Subset construction
    -------------------
    • Draw exactly 500 images per class (5000 total) from the 50 000-image
      training split using your personal seed (get_seed()).
    • Sampling must be without replacement within each class.
    • Use the same seed for every class so the selection is deterministic.

    Normalisation
    -------------
    Both train and test use the following transform pipeline:
        ToTensor()
        Normalize(mean=(0.4914, 0.4822, 0.4465),
                  std =(0.2470, 0.2435, 0.2616))

    These are the standard per-channel statistics for CIFAR-10.

    Parameters
    ----------
    data_root : str – Directory where CIFAR-10 will be downloaded / cached.

    Returns
    -------
    train_subset : torch.utils.data.Subset
        Subset of the training split containing exactly 5000 images
        (500 per class), selected using your personal seed.
    test_dataset : torchvision.datasets.CIFAR10
        The full 10 000-image test split.

    Hint
    ----
    After loading the full training dataset, collect the indices for each
    class label using a loop or list comprehension, then use
    random.sample(class_indices, 500) with the seed set beforehand.
    Concatenate all class index lists and wrap in a Subset.
    """
    # TODO 1.5 ── Implement this function.
    #
    #   1. Define the transform (ToTensor + Normalize as above).
    #   2. Load the full CIFAR-10 train and test splits.
    #   3. call set_all_seeds(get_seed()) before any sampling.
    #   4. For each class 0-9, collect its indices in the training set,
    #      then randomly sample 500 of them.
    #   5. Return (Subset(train_dataset, selected_indices), test_dataset).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std =(0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    
    set_all_seeds(get_seed())
    targets = train_dataset.targets
    selected_indices: List[int] = []
    for cls in range(10):
        cls_indices = [i for i, t in enumerate(targets) if t == cls]
        sampled = random.sample(cls_indices, 500)
        selected_indices.extend(sampled)
    
    return Subset(train_dataset, selected_indices), test_dataset


def train_model(
    model:             VisionTransformer,
    train_subset:      Subset,
    test_dataset:      datasets.CIFAR10,
    config:            Dict,
    checkpoint_dir:    str             = "checkpoints",
    checkpoint_epochs: Tuple[int, ...] = CHECKPOINT_EPOCHS,
    log_path:          Optional[str]   = "training_log.json",
) -> Dict:
    """
    Train the model for config["epochs"] epochs and save checkpoints.

    Optimiser & schedule
    --------------------
    • Optimiser : AdamW with lr = config["lr"], weight_decay = 1e-4.
    • LR schedule: CosineAnnealingLR(optimizer, T_max=config["epochs"]).
      Call scheduler.step() once per epoch (after the epoch loop, not per batch).

    Per-epoch logging
    -----------------
    After each epoch record:
        epoch          : int   (1-indexed)
        train_loss     : float (mean cross-entropy loss over all training batches)
        val_accuracy   : float (fraction of test images correctly classified)
        epoch_time_sec : float (wall-clock seconds for that epoch)

    Checkpointing
    -------------
    At each epoch listed in checkpoint_epochs, save:
        torch.save({
            "model_state_dict": model.state_dict(),
            "config":           model.config,
            "epoch":            epoch,
            "student_id":       STUDENT_ID,
        }, f"{checkpoint_dir}/baseline_epoch_{epoch}.pt")

    JSON log
    --------
    Write the log to log_path (if not None) with the schema:
        {
          "student_id":         STUDENT_ID,
          "seed":               get_seed(),
          "config":             config,          ← full training config dict
          "history":            [ {epoch, train_loss, val_accuracy,
                                   epoch_time_sec}, ... ],
          "final_val_accuracy": float,           ← history[-1]["val_accuracy"]
          "total_params":       int              ← total trainable parameters
        }

    All float values must be rounded to 4 decimal places.

    Parameters
    ----------
    model             : VisionTransformer – The model to train (CPU).
    train_subset      : Subset            – Training data from get_cifar10_subset.
    test_dataset      : CIFAR10           – Full test set.
    config            : dict              – Training hyper-parameters.
    checkpoint_dir    : str               – Directory for .pt files.
    checkpoint_epochs : tuple of int      – Epochs at which to save checkpoints.
    log_path          : str or None       – Path for the JSON log file.

    Returns
    -------
    log : dict  – The same dictionary that was (optionally) written to disk.

    Hints
    -----
    • Create DataLoaders inside this function:
          train_loader = DataLoader(train_subset, batch_size=config["batch_size"],
                                    shuffle=True)
          test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False)
    • Use model.train() / model.eval() around the training and validation loops.
    • Count total trainable parameters with:
          sum(p.numel() for p in model.parameters() if p.requires_grad)
    • Use time.time() to measure epoch wall-clock time.
    • Create checkpoint_dir if it doesn't exist: os.makedirs(..., exist_ok=True)
    """
    # TODO 1.6 ── Implement the training loop.
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_loader = DataLoader(train_subset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history: List[Dict] = []
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        model.train()
        epoch_losses: List[float] = []
        
        for images, labels in train_loader:
            logits, _ = model(images)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                logits, _ = model(images)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_time = time.time() - t0
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        val_accuracy = correct / total if total > 0 else 0.0

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(val_accuracy, 4),
            "epoch_time_sec": round(epoch_time, 4),
        }
        history.append(entry)

        if epoch in checkpoint_epochs:
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": model.config,
                "epoch": epoch,
                "student_id": STUDENT_ID,
            }
            torch.save(ckpt, os.path.join(checkpoint_dir, f"baseline_epoch_{epoch}.pt"))

    log = {
        "student_id": STUDENT_ID,
        "seed": get_seed(),
        "config": config,
        "history": history,
        "final_val_accuracy": round(history[-1]["val_accuracy"], 4) if history else 0.0,
        "total_params": total_params,
    }

    if log_path is not None:
        _save_json(log, log_path)

    return log


# =============================================================================
# PART 2 – Ablation Studies
# =============================================================================

def run_ablations(ablation_dir: str = "ablation_logs") -> None:
    """
    Train all 7 ablation models and save their logs.

    Each run changes exactly ONE hyper-parameter from the baseline and keeps
    all others fixed.  Use the baseline CHECKPOINT_EPOCHS tuple for saving
    checkpoints but store them in a subdirectory so they don't overwrite the
    baseline checkpoints.

    Ablations
    ---------
    2.1  Patch size  : train with patch_size ∈ {2, 4, 8}
         Log files   : ablation_logs/patch_2.json
                       ablation_logs/patch_4.json
                       ablation_logs/patch_8.json

    2.2  Num layers  : train with num_layers ∈ {2, 4, 8}
         Log files   : ablation_logs/layers_2.json
                       ablation_logs/layers_4.json
                       ablation_logs/layers_8.json

    2.3  No pos enc  : baseline config but pos_embed multiplied by 0
                       (see _zero_pos_embed helper below)
         Log file    : ablation_logs/no_pos_enc.json

    Important
    ---------
    • Call set_all_seeds(get_seed()) before each model construction so that
      weight initialisation is the same across runs (only the architecture
      or the forward pass changes).
    • Each ablation run uses a fresh call to get_cifar10_subset() — this
      ensures the same data subset for all runs.
    • Save checkpoints into ablation_checkpoints/<run_name>/ to avoid
      overwriting the baseline .pt files.
    """
    os.makedirs(ablation_dir, exist_ok=True)

    seed = get_seed()
    train_subset, test_dataset = get_cifar10_subset()

    # ── 2.1  Patch size ──────────────────────────────────────────────────────
    for patch_size in [2, 4, 8]:
        print(f"\n[Ablation] patch_size={patch_size}")
        cfg = {**BASELINE_CONFIG, "patch_size": patch_size}
        set_all_seeds(seed)
        model = build_model(cfg)
        ckpt_dir = f"ablation_checkpoints/patch_{patch_size}"
        train_model(
            model, train_subset, test_dataset, cfg,
            checkpoint_dir=ckpt_dir, checkpoint_epochs=(),
            log_path=os.path.join(ablation_dir, f"patch_{patch_size}.json"),
        )

    # ── 2.2  Number of layers ─────────────────────────────────────────────────
    for num_layers in [2, 4, 8]:
        print(f"\n[Ablation] num_layers={num_layers}")
        cfg = {**BASELINE_CONFIG, "num_layers": num_layers}
        set_all_seeds(seed)
        model = build_model(cfg)
        ckpt_dir = f"ablation_checkpoints/layers_{num_layers}"
        train_model(
            model, train_subset, test_dataset, cfg,
            checkpoint_dir=ckpt_dir, checkpoint_epochs=(),
            log_path=os.path.join(ablation_dir, f"layers_{num_layers}.json"),
        )

    # ── 2.3  No positional encoding ───────────────────────────────────────────
    print("\n[Ablation] no_pos_enc")
    cfg = {**BASELINE_CONFIG}
    set_all_seeds(seed)
    model = build_model(cfg)
    _zero_pos_embed(model)          # disable positional embeddings
    ckpt_dir = "ablation_checkpoints/no_pos_enc"
    train_model(
        model, train_subset, test_dataset, cfg,
        checkpoint_dir=ckpt_dir, checkpoint_epochs=(),
        log_path=os.path.join(ablation_dir, "no_pos_enc.json"),
    )


def _zero_pos_embed(model: VisionTransformer) -> None:
    """
    Permanently zero out the positional embedding parameter so that it has
    no effect during the forward pass.

    After this call the model behaves identically to one where the line
        x = x + pos_embed
    is replaced by
        x = x  (i.e., the addition is skipped).

    We achieve this by zeroing the data AND detaching / freezing the
    parameter so gradient updates cannot restore the signal.

    Parameters
    ----------
    model : VisionTransformer – Modified in-place.
    """
    with torch.no_grad():
        model.pos_embed.data.zero_()
    model.pos_embed.requires_grad = False


# =============================================================================
# PART 3 – Quantitative Analysis
# =============================================================================

def _load_baseline_checkpoint(
    checkpoint_path: str = "checkpoints/baseline_epoch_20.pt",
) -> VisionTransformer:
    """
    Load the baseline checkpoint and return the model in eval mode.

    The config dict stored inside the checkpoint is used to reconstruct the
    model architecture, so the function does not need any hyper-parameters
    passed explicitly.

    Parameters
    ----------
    checkpoint_path : str – Path to the .pt file.

    Returns
    -------
    model : VisionTransformer  (CPU, eval mode)
    """
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def compute_attention_entropy(
    checkpoint_path: str = "checkpoints/baseline_epoch_20.pt",
    output_path:     str = "analysis/attention_entropy.json",
) -> Dict[str, float]:
    """
    Compute the mean Shannon entropy of the CLS token's attention
    distribution for each Transformer layer.

    Definition
    ----------
    For a single attention distribution p (a probability vector of length T):

        H(p) = -∑_i  p_i · log2(p_i)

    where p_i is clamped to [ε, 1] with ε = 1e-9 before taking the log.

    Averaging
    ---------
    For each layer L:
        1. Collect CLS attention weights across ALL test images.
           Shape per image: (h, T, T) → take row 0 → (h, T).
        2. Average over heads and over images.
        3. Compute entropy of the resulting mean distribution.

    Note: "row 0" of the attention matrix (h, T, T) corresponds to the
    attention weights FROM the CLS token TO all tokens (including itself).

    Parameters
    ----------
    checkpoint_path : str – Path to the baseline .pt checkpoint.
    output_path     : str – Where to write the result JSON.

    Returns
    -------
    result : dict  e.g. {"layer_0": 3.2114, "layer_1": 2.8532, ...}

    Hints
    -----
    • Use _load_baseline_checkpoint to load the model.
    • Use get_cifar10_subset() to get the test loader:
          _, test_dataset = get_cifar10_subset()
          test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    • With torch.no_grad(), run a forward pass:
          logits, attn_list = model(images)
          # attn_list is a list of L tensors, each (B, h, T, T)
    • Accumulate the per-layer CLS attentions across batches.
    • Remember to call set_all_seeds(get_seed()) for reproducibility even
      though this function does not involve randomness.
    • os.makedirs(os.path.dirname(output_path), exist_ok=True) before writing.
    """
    # TODO 3.1 -- Implement attention entropy computation.
    raise NotImplementedError("TODO 3.1: implement compute_attention_entropy")


def compute_pos_embed_correlation(
    checkpoint_path: str = "checkpoints/baseline_epoch_20.pt",
    output_path:     str = "analysis/pos_embed_correlation.json",
) -> Dict[str, float]:
    """
    Measure whether the learned positional embeddings capture spatial proximity.

    Method
    ------
    1. Extract pos_embed of shape (1, N+1, D).  Discard index 0 (CLS) to
       obtain patch embeddings of shape (N, D).
    2. Compute the N×N pairwise cosine similarity matrix S.
    3. Assign 2-D grid coordinates to each patch index k:
           row = k // G,   col = k % G,   where G = 32 // patch_size
    4. Compute the N×N pairwise Euclidean distance matrix E between
       these 2-D coordinates.
    5. Extract the STRICT upper triangle (diagonal excluded) from both S
       and E as 1-D vectors.
    6. Compute Pearson correlation r between the two vectors.

    Expected result
    ---------------
    r should be negative (spatially closer patches → higher cosine similarity
    → smaller distance, so similarity and distance are anti-correlated).

    Parameters
    ----------
    checkpoint_path : str – Path to the baseline .pt checkpoint.
    output_path     : str – Where to write the result JSON.

    Returns
    -------
    result : dict  e.g. {"pearson_r": -0.4213, "num_pairs": 2016}

    Hints
    -----
    • F.normalize(embeddings, dim=-1) then (normed @ normed.T) gives cosine
      similarity directly.
    • For pairwise Euclidean distance, broadcast:
          coords  shape (N, 2)
          diff    = coords[:, None, :] - coords[None, :, :]   → (N, N, 2)
          E       = diff.norm(dim=-1)
    • torch.triu_indices(N, N, offset=1) gives the upper-triangle indices.
    • np.corrcoef(a, b)[0, 1] computes Pearson r.
    • num_pairs = N * (N - 1) // 2
    """
    # TODO 3.2 -- Implement positional embedding correlation.
    raise NotImplementedError("TODO 3.2: implement compute_pos_embed_correlation")


def compute_per_class_accuracy(
    checkpoint_path: str = "checkpoints/baseline_epoch_20.pt",
    output_path:     str = "analysis/per_class_accuracy.json",
) -> Dict:
    """
    Evaluate the model on the full test set and report per-class accuracy
    and the top-3 most frequent misclassification pairs.

    Method
    ------
    1. Run the model on the full test set (no_grad).
    2. For each class c ∈ {0 … 9}:
           accuracy_c = (# test images of class c predicted as c) / (# images of class c)
    3. Build a confusion count table:
           misclassifications[true_label][pred_label] += 1
       (only for cases where true_label ≠ pred_label).
    4. Sort (true, pred, count) triples by count descending, take top 3.

    Parameters
    ----------
    checkpoint_path : str – Path to the baseline .pt checkpoint.
    output_path     : str – Where to write the result JSON.

    Returns
    -------
    result : dict with keys:
        "class_accuracies" : {"0": float, "1": float, ..., "9": float}
        "top3_confusions"  : [[true, pred, count], ...] (3 entries)

    Hints
    -----
    • Accumulate (true_labels, pred_labels) tensors across batches, then
      torch.cat them before computing stats.
    • A 10×10 confusion matrix is the cleanest approach:
          conf = torch.zeros(10, 10, dtype=torch.long)
          for t, p in zip(true_labels, pred_labels):
              conf[t, p] += 1
      Diagonal entries are correct predictions; off-diagonal are confusions.
    """
    # TODO 3.3 -- Implement per-class accuracy and confusion analysis.
    raise NotImplementedError("TODO 3.3: implement compute_per_class_accuracy")


def compute_attention_distance(
    checkpoint_path: str = "checkpoints/baseline_epoch_20.pt",
    output_path:     str = "analysis/attention_distance.json",
) -> Dict[str, float]:
    """
    Compute the mean attention distance for patch tokens in each layer.

    This metric quantifies how far apart the patches are that each query patch
    attends to, providing a measure of the receptive field of attention.

    Method (per head, per layer, per image)
    ----------------------------------------
    Let A ∈ ℝ^{(N+1)×(N+1)} be the full attention weight matrix.
    Extract A' = A[1:, 1:]  ∈ ℝ^{N×N}  (patch-to-patch block, excluding CLS).
    Row-normalise A' so each row sums to 1 (use clamp + divide to avoid /0).

    Assign 2-D grid coordinates to each patch index k:
        row_k = k // G,   col_k = k % G,   G = 32 // patch_size

    For each query patch i:
        d_i = ∑_j  A'[i, j] · sqrt( (row_i - row_j)² + (col_i - col_j)² )

    Mean distance for this head = mean over i of d_i.

    Final value for layer L:
        average over all heads, then over all test images.

    Parameters
    ----------
    checkpoint_path : str – Path to the baseline .pt checkpoint.
    output_path     : str – Where to write the result JSON.

    Returns
    -------
    result : dict  e.g. {"layer_0": 2.3112, "layer_1": 2.5534, ...}

    Hints
    -----
    • Pre-compute the (N, 2) coordinate array and pairwise distance matrix
      D_grid ∈ ℝ^{N×N} once before the data loop.
    • attn_list[l] has shape (B, h, T, T).  Slice [:, :, 1:, 1:] to get the
      (B, h, N, N) patch-to-patch block.
    • Row-normalise: A_patch = A_patch / A_patch.sum(dim=-1, keepdim=True).clamp(min=1e-9)
    • Broadcast the pre-computed D_grid (shape N×N) against A_patch (B, h, N, N):
          mean_dist_per_query = (A_patch * D_grid).sum(dim=-1)  → (B, h, N)
      Then .mean(dim=-1) → (B, h), then .mean() for the scalar.
    """
    # TODO 3.4 -- Implement mean attention distance computation.
    raise NotImplementedError("TODO 3.4: implement compute_attention_distance")


# =============================================================================
# Utility: write a dict to a JSON file with 4-decimal rounding
# =============================================================================

def _save_json(data: Dict, path: str) -> None:
    """
    Recursively round all float values to 4 decimal places and write to path.

    Creates parent directories as needed.
    """
    def _round(obj):
        if isinstance(obj, float):
            return round(obj, 4)
        if isinstance(obj, dict):
            return {k: _round(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_round(data), f, indent=2)
    print(f"  Saved → {path}")


# =============================================================================
# Runner
# =============================================================================

def run_baseline() -> None:
    """Train the baseline model and produce checkpoints + training log."""
    print("=" * 60)
    print("BASELINE TRAINING")
    print("=" * 60)
    seed = get_seed()
    print(f"  STUDENT_ID : {STUDENT_ID}")
    print(f"  Seed       : {seed}")

    set_all_seeds(seed)
    train_subset, test_dataset = get_cifar10_subset()
    print(f"  Train size : {len(train_subset)}")
    print(f"  Test  size : {len(test_dataset)}")

    set_all_seeds(seed)
    model = build_model(BASELINE_CONFIG)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters : {total_params:,}")

    train_model(
        model            = model,
        train_subset     = train_subset,
        test_dataset     = test_dataset,
        config           = BASELINE_CONFIG,
        checkpoint_dir   = "checkpoints",
        checkpoint_epochs= CHECKPOINT_EPOCHS,
        log_path         = "training_log.json",
    )


def run_analysis() -> None:
    """Run all four analysis functions on the baseline checkpoint."""
    print("=" * 60)
    print("QUANTITATIVE ANALYSIS")
    print("=" * 60)
    os.makedirs("analysis", exist_ok=True)

    print("\n[3.1] Attention Entropy")
    compute_attention_entropy()

    print("\n[3.2] Positional Embedding Distance Correlation")
    compute_pos_embed_correlation()

    print("\n[3.3] Per-Class Accuracy and Top-3 Confusions")
    compute_per_class_accuracy()

    print("\n[3.4] Mean Attention Distance")
    compute_attention_distance()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CSE676 Assignment 1 – Vision Transformers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes
-----
  baseline   Train the baseline ViT only.
  all        Train baseline → run ablations → run analysis (recommended).
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "all"],
        default="all",
        help="Which part to run (default: all).",
    )
    args = parser.parse_args()

    # Validate student ID early so the error is obvious.
    _ = get_seed()

    run_baseline()

    if args.mode == "all":
        print("\n" + "=" * 60)
        print("ABLATION STUDIES")
        print("=" * 60)
        run_ablations()

        run_analysis()

    print("\nDone.  Inspect the JSON files in ablation_logs/ and analysis/,")
    print("then write your report.  Remember to run check_git.py before submitting!")


if __name__ == "__main__":
    main()

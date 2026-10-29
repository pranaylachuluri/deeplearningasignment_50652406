# Vision Transformer from Scratch (CIFAR-10)

A Vision Transformer (ViT) written from scratch in PyTorch and trained on a class-balanced 5,000-image subset of CIFAR-10. It includes ablation studies on depth, patch size and positional encoding, plus an analysis of what the attention heads learn.

Built for CSE 676 Deep Learning at the University at Buffalo (Spring 2026).

## What's implemented

- **Patch embedding:** images are split into non-overlapping patches and projected to tokens, with a learnable `[CLS]` token
- **Learnable positional embeddings** (they can be switched off for the ablation)
- **Transformer encoder:** multi-head self-attention, a feed-forward MLP, LayerNorm and residual connections
- **Training pipeline:** seeded data subset, AdamW with a cosine learning-rate schedule, accuracy logging and checkpoints at epochs 5, 10 and 20
- **Analysis:** per-class accuracy, top confusions, mean attention distance, attention entropy and positional-embedding similarity

## Results

The baseline uses 4×4 patches, 4 layers, 2 heads and a 64-dim embedding, trained for 20 epochs (142K parameters). Accuracy is measured on the full CIFAR-10 test set.

| Experiment | Parameters | Test accuracy |
| --- | ---: | ---: |
| **Baseline** (patch 4, 4 layers) | 142K | **48.4%** |
| Patch size 2 | 152K | 49.3% |
| Patch size 8 | 148K | 40.4% |
| 2 layers | 75K | 48.5% |
| 8 layers | 276K | 47.1% |
| No positional encoding | 138K | 45.2% |

**What the ablations show**

- Smaller patches help: 8×8 patches leave only 16 tokens for a 32×32 image, and accuracy drops 8 points.
- With only 5,000 training images, more depth does not help. The 8-layer model is slightly worse than the 2-layer one.
- Removing positional encoding costs about 3 points, so the model does use spatial layout.
- The most confused pairs are cat ↔ dog and truck ↔ automobile.

## Repository layout

| Path | Contents |
| --- | --- |
| `vit_template.py` | Model, training loop, ablations and analysis |
| `training_log.json` | Baseline loss and accuracy per epoch |
| `ablation_logs/` | One JSON log per ablation run |
| `analysis/` | Attention distance and entropy, per-class accuracy, positional-embedding correlation |
| `checkpoints/` | Baseline weights at epochs 5, 10 and 20 |

## Run it

```bash
pip install -r requirements.txt
python vit_template.py --mode all
```

CIFAR-10 downloads automatically into `./data`. A full run (baseline, ablations and analysis) takes a few minutes on a GPU.

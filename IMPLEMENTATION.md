# Implementation Specification: Bridge Anchors

This document is the detailed implementation spec. When implementing, follow this spec closely.

---

## 1. Core Model: BridgeAnchorAligner

### Architecture

```python
class BridgeAnchorAligner(nn.Module):
    """
    Aligns two independently trained encoder spaces using learnable bridge anchors.
    
    Each embedding is converted to a K-dimensional vector of cosine similarities
    to K learnable anchor points. The resulting "bridged" representations live in
    the same K-dimensional space and can be directly compared.
    
    Args:
        dim_img: Dimension of image encoder output (e.g., 768 for DINOv2 ViT-B)
        dim_txt: Dimension of text encoder output (e.g., 768 for all-mpnet-base-v2)
        num_anchors: Number of anchor points K (default: 32)
        init_method: 'random' | 'prototype' — anchor initialization strategy
    """
    
    # Learnable parameters:
    #   anchors_img: (K, dim_img) — anchor positions in image space
    #   anchors_txt: (K, dim_txt) — anchor positions in text space
    
    # Forward pass:
    #   Input:  img_emb (B, dim_img), txt_emb (B, dim_txt)
    #   Step 1: Normalize anchors to unit sphere
    #   Step 2: b_img = normalize(img_emb @ anchors_img.T)  # (B, K)
    #   Step 3: b_txt = normalize(txt_emb @ anchors_txt.T)  # (B, K)
    #   Output: b_img (B, K), b_txt (B, K)
```

### Key implementation details

- Anchors are `nn.Parameter`, initialized with `nn.init.normal_` then L2-normalized
- For 'prototype' init: compute class-mean embeddings from training data, use as initial anchors
- Both anchors and input embeddings should be L2-normalized before computing cosine similarity
- The output bridged representations should also be L2-normalized

### Shapes (example with B=64, dim_img=768, dim_txt=768, K=32)
```
img_emb:      (64, 768)
txt_emb:      (64, 768)  
anchors_img:  (32, 768)  ← nn.Parameter, learnable
anchors_txt:  (32, 768)  ← nn.Parameter, learnable
b_img:        (64, 32)   ← output, normalized
b_txt:        (64, 32)   ← output, normalized
```

---

## 2. Baselines

### LinearProjection
```python
class LinearProjection(nn.Module):
    # Projects image embeddings into text embedding space
    # proj: nn.Linear(dim_img, dim_txt, bias=False)
    # Forward: normalize(proj(img_emb)), normalize(txt_emb)
    # Params: dim_img * dim_txt (e.g., 768*768 = 590K)
```

### MLPProjection
```python
class MLPProjection(nn.Module):
    # 2-layer MLP with bottleneck: dim_img -> hidden -> dim_txt
    # hidden_dim = 256 by default
    # Forward: normalize(mlp(img_emb)), normalize(txt_emb)
    # Params: dim_img*hidden + hidden*dim_txt (e.g., 768*256 + 256*768 ≈ 400K)
```

### FixedRelativeRep
```python
class FixedRelativeRep(nn.Module):
    # Moschella et al. baseline — no learnable parameters
    # Select K paired samples from training data as anchors
    # fixed_anchors_img: (K, dim_img) — registered buffer, NOT parameter
    # fixed_anchors_txt: (K, dim_txt) — registered buffer, NOT parameter
    # Forward: same cosine similarity computation as BridgeAnchors
    # but anchors are fixed (no gradient)
    # Params: 0
```

---

## 3. Loss Function

### InfoNCE Loss
```python
def info_nce_loss(img_features, txt_features, temperature=0.07):
    """
    Symmetric InfoNCE loss for cross-modal alignment.
    
    Args:
        img_features: (B, D) normalized image representations
        txt_features: (B, D) normalized text representations
        temperature: scalar temperature for logits
    
    Returns:
        loss: scalar, average of img->txt and txt->img losses
    
    Implementation:
        logits = img_features @ txt_features.T / temperature  # (B, B)
        labels = torch.arange(B)  # diagonal is positive
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        loss = (loss_i2t + loss_t2i) / 2
    """
```

---

## 4. Data Pipeline

### Embedding Extraction (run once, save to disk)

```python
# extract_embeddings.py

# Image encoder: DINOv2 ViT-B/14
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
# Use CLS token output: (B, 768)

# Text encoder: all-mpnet-base-v2  
from sentence_transformers import SentenceTransformer
text_model = SentenceTransformer('all-mpnet-base-v2')
# Output: (B, 768)

# For COCO: extract and save as .pt files
# img_embeddings.pt: (118287, 768)  — all COCO train images
# txt_embeddings.pt: (118287, 768)  — corresponding captions (one per image)
# For multiple captions per image, use the first caption for simplicity

# For Flickr30k test: similar extraction
# For ImageNet val: extract image embeddings + generate text embeddings for all 1000 class names
```

### Training Dataset
```python
class PairedEmbeddingDataset(Dataset):
    """
    Loads pre-extracted embedding pairs from .pt files.
    Supports subsampling for data efficiency experiments.
    
    Args:
        img_emb_path: path to image embeddings .pt file
        txt_emb_path: path to text embeddings .pt file
        num_samples: if set, randomly subsample this many pairs
    """
```

---

## 5. Training Loop

```python
# Pseudocode for training

config = {
    'num_anchors': 32,
    'batch_size': 256,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'epochs': 20,
    'temperature': 0.07,
    'scheduler': 'cosine',
    'warmup_epochs': 2,
}

model = BridgeAnchorAligner(dim_img=768, dim_txt=768, num_anchors=32)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=20)

for epoch in range(20):
    for img_emb, txt_emb in dataloader:
        b_img, b_txt = model(img_emb, txt_emb)
        loss = info_nce_loss(b_img, b_txt, temperature=0.07)
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Evaluate every epoch on Flickr30k
    metrics = evaluate_retrieval(model, flickr_loader)
    log(epoch, loss, metrics)

# Save checkpoint
torch.save(model.state_dict(), 'checkpoint.pt')
```

---

## 6. Evaluation

### Retrieval (Flickr30k)
```python
def evaluate_retrieval(model, img_embs, txt_embs):
    """
    Compute image-to-text and text-to-image retrieval metrics.
    
    Steps:
        1. Get bridged representations: b_img, b_txt = model(img_embs, txt_embs)
        2. Compute similarity matrix: sims = b_img @ b_txt.T  # (N, N)
        3. For each image, rank all texts by similarity → compute R@1, R@5, R@10
        4. For each text, rank all images by similarity → compute R@1, R@5, R@10
    
    Returns:
        dict with keys: i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10
    """
```

### Zero-shot Classification (ImageNet)
```python
def evaluate_zeroshot(model, img_embs, class_names):
    """
    Zero-shot classification using text prompts.
    
    Steps:
        1. Generate text embeddings for "a photo of a {class}" for all 1000 classes
        2. Get bridged representations for all class texts and test images
        3. Predict class = argmax cosine_similarity(b_img, b_class_txt)
        4. Compute top-1 accuracy
    """
```

### Anchor Analysis (Direction A)
```python
def analyze_anchors(model, img_embs, txt_embs, labels=None):
    """
    Analyze what the learned anchors represent.
    
    Analysis 1: Nearest neighbors
        - For each image anchor, find the K nearest training images
        - For each text anchor, find the K nearest training texts
        - Check if corresponding image/text anchors point to similar concepts
    
    Analysis 2: Anchor similarity structure
        - Compute anchor-anchor similarity in image space: A_img @ A_img.T
        - Compute anchor-anchor similarity in text space: A_txt @ A_txt.T
        - Compare the two matrices (should be similar if alignment is working)
    
    Analysis 3: Class alignment (if labels available)
        - For each anchor, compute which class's images/texts are closest
        - Visualize anchor-class correspondence
    """
```

---

## 7. Experiments

### Experiment A: Main comparison
- Train: BridgeAnchors(K=32), LinearProjection, MLPProjection, FixedRelativeRep(K=32)
- All on COCO 118K, evaluate on Flickr30k + ImageNet
- 3 random seeds each

### Experiment B: K ablation
- K = {4, 8, 16, 32, 64, 128, 256}
- BridgeAnchors only, COCO 118K, Flickr30k eval

### Experiment C: Data efficiency
- Training samples = {500, 1K, 5K, 10K, 50K, 118K}
- BridgeAnchors(K=32) vs LinearProjection
- Flickr30k eval

### Experiment D: Fixed vs Learnable anchors
- Same K=32, four strategies:
  (1) Fixed random, (2) Fixed prototype, (3) Learnable random, (4) Learnable prototype
- COCO 118K, Flickr30k eval

---

## 8. Implementation Priority

1. **First**: `extract_embeddings.py` — get embeddings saved to disk
2. **Second**: `bridge_anchors.py` + `losses.py` + training loop — core model
3. **Third**: `retrieval.py` — basic evaluation
4. **Fourth**: `baselines.py` — comparison models
5. **Fifth**: `anchor_analysis.py` — Direction A analysis
6. **Last**: Experiment scripts for B, C, D

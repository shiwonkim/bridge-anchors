# Datasets

This directory contains (or symlinks to) the datasets used for training and evaluation.

## Flickr30k

**Purpose:** Image-text retrieval evaluation (R@1, R@5, R@10).

### Required files

```
flickr30k/
├── flickr30k_images/          # 31,783 JPEG images
│   ├── 1000092795.jpg
│   ├── 1000341163.jpg
│   └── ...
└── results_20130124.token     # Captions file (tab-separated)
```

### How to obtain

Flickr30k requires manual download. Two options:

#### Option 1: Kaggle (recommended)
1. Go to: https://www.kaggle.com/datasets/eeshawn/flickr30k
2. Download and extract `flickr30k_images.zip`
3. Download `results_20130124.token` (captions file)

#### Option 2: Official request
1. Fill the request form at: https://forms.illinois.edu/sec/229675
2. You will receive a download link via email

### After downloading

Place files so the directory looks like the structure above, then run:

```bash
python src/data/extract_embeddings.py --dataset flickr30k
```

The extraction script auto-detects common layouts:
- `flickr30k/images/`
- `flickr30k/flickr30k_images/`
- `flickr30k/flickr30k-images/`

---

## ImageNet ILSVRC2012 Validation Set

**Purpose:** Zero-shot classification evaluation.

### Required files

```
imagenet/
├── val/                              # 50,000 images in 1,000 class subdirectories
│   ├── n01440764/
│   │   ├── ILSVRC2012_val_00000293.JPEG
│   │   └── ...
│   ├── n01443537/
│   └── ...
└── imagenet_classes.txt              # Optional: 1,000 class names (one per line)
```

### How to obtain

ImageNet requires academic access:

1. Register at: https://image-net.org/signup
2. Request access to ILSVRC2012
3. Download `ILSVRC2012_img_val.tar` (~6.3 GB)

### After downloading

```bash
# Extract validation images
mkdir -p val && tar xf ILSVRC2012_img_val.tar -C val/

# Organize into class subdirectories (requires the devkit)
cd val && wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd ..
```

Then run embedding extraction:

```bash
python src/data/extract_embeddings.py --dataset imagenet
```

### Alternative layouts

The extraction script also supports a flat layout with a ground truth file:

```
imagenet/
├── val/                              # 50,000 JPEG files (no subdirectories)
└── ILSVRC2012_val_ground_truth.txt   # One integer label per line
```

### Note on tiny-imagenet

A `tiny-imagenet-200` dataset exists on this machine at `/mnt/2021_NIA_data/tiny-imagenet-200/`
but it only has 200 classes (vs 1,000) and lower-resolution images, so it's not suitable
for standard ImageNet zero-shot evaluation.

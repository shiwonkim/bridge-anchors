#!/bin/bash
# Extract embeddings from frozen encoders and save as .pt files

set -e

echo "Extracting COCO train embeddings..."
python src/data/extract_embeddings.py --dataset coco --split train

echo "Extracting Flickr30k test embeddings..."
python src/data/extract_embeddings.py --dataset flickr30k --split test

echo "Extracting ImageNet val embeddings..."
python src/data/extract_embeddings.py --dataset imagenet --split val

echo "Done."

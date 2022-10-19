#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# Download command: bash ./scripts/get_coco.sh

Download/unzip labels
cd data # unzip directory
url=https://drive.google.com/uc?id=1BKMXnyPFT6uFWCSbyrZ7r5st9bCQwb2T
echo 'Downloading dataset'
gdown https://drive.google.com/uc?id=1BKMXnyPFT6uFWCSbyrZ7r5st9bCQwb2T
unzip dataset.zip
rm dataset.zip
cd ..
echo 'Making splits'
python scripts/make_split.py \
    --dataset_path data/dataset \
    --valid_size 300 \
    --test_size 200 \

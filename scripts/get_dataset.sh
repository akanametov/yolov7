#!/usr/bin/env bash

echo "Donwloading dataset ..."

wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${1:-1BKMXnyPFT6uFWCSbyrZ7r5st9bCQwb2T}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${1:-1BKMXnyPFT6uFWCSbyrZ7r5st9bCQwb2T}" -O data/dataset.zip && unzip data/dataset.zip -d data && rm -rf /tmp/cookies.txt 

echo 'Making splits ...'

python scripts/make_split.py \
    --dataset_path data/dataset \
    --valid_size 300 \
    --test_size 200 \

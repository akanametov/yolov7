import argparse

import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Make splits')
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument('--valid_size', default=300, type=int)
    parser.add_argument('--test_size', default=200, type=int)
    args = parser.parse_args()

    images = sorted([f for f in os.listdir(args.dataset_path) if f.endswith(".jpg")])
    annotations = sorted([f for f in os.listdir(args.dataset_path) if f.endswith(".txt")])
    print("Number of images", len(images))
    print("Number of annotations", len(annotations))

    train_images, valid_images, train_annotations, valid_annotations = train_test_split(
        images,
        annotations,
        test_size=args.valid_size + args.test_size,
        random_state=1,
    )
    valid_images, test_images, valid_annotations, test_annotations = train_test_split(
        valid_images,
        valid_annotations,
        test_size=args.test_size,
        random_state=1,
    )

    SPLIT_FILES = {
        "train": (train_images, train_annotations),
        "valid": (valid_images, valid_annotations),
        "test": (test_images, test_annotations),
    }
    for split in SPLIT_FILES.keys():
        print("Split", split)
        os.makedirs(os.path.join(args.dataset_path, split), exist_ok=True)
        for image_name, anno_name in tqdm(zip(*SPLIT_FILES[split])):
            # copy image
            shutil.copy(os.path.join(args.dataset_path, image_name), os.path.join(args.dataset_path, split, image_name))
            # copy annotation
            shutil.copy(os.path.join(args.dataset_path, anno_name), os.path.join(args.dataset_path, split, anno_name))
    print("Done!")

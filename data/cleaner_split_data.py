import os
import shutil
import random
from pathlib import Path

source_classes_dir = Path("data/classes")
source_nobird_dir = Path("data/no_bird")
target_base = Path("data/data_cleaner/binary_dataset")
train_ratio = 0.8

# Setup target folders
for split in ['train', 'val']:
    for cls in ['bird', 'no_bird']:
        os.makedirs(target_base / split / cls, exist_ok=True)

# Collect and split bird images
bird_images = list(source_classes_dir.glob("*/*.jpg"))
random.shuffle(bird_images)
split_idx = int(len(bird_images) * train_ratio)
train_birds = bird_images[:split_idx]
val_birds = bird_images[split_idx:]

# Collect and split no_bird images
nobird_images = list(source_nobird_dir.glob("*.jpg"))
random.shuffle(nobird_images)
split_idx = int(len(nobird_images) * train_ratio)
train_nobirds = nobird_images[:split_idx]
val_nobirds = nobird_images[split_idx:]

# Copy files
for src in train_birds:
    shutil.copy(src, target_base / "train" / "bird" / src.name)
for src in val_birds:
    shutil.copy(src, target_base / "val" / "bird" / src.name)
for src in train_nobirds:
    shutil.copy(src, target_base / "train" / "no_bird" / src.name)
for src in val_nobirds:
    shutil.copy(src, target_base / "val" / "no_bird" / src.name)
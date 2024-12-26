import os
import shutil
import random

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8):
    train_images_dir = os.path.join(output_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_dir, 'train', 'labels')
    val_images_dir = os.path.join(output_dir, 'val', 'images')
    val_labels_dir = os.path.join(output_dir, 'val', 'labels')
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    image_files = {f.replace('.jpeg', '').replace('.png', ''): f for f in os.listdir(images_dir) if f.endswith(('.jpeg', '.png'))}
    label_files = {f.replace('.txt', ''): f for f in os.listdir(labels_dir) if f.endswith('.txt')}

    # Find matching pairs of images and labels
    matched_files = [(image_files[base], label_files[base]) for base in image_files if base in label_files]

    random.shuffle(matched_files)

    split_index = int(len(matched_files) * train_ratio)
    train_files = matched_files[:split_index]
    val_files = matched_files[split_index:]

    for image_file, label_file in train_files:
        shutil.copy(os.path.join(images_dir, image_file), train_images_dir)
        shutil.copy(os.path.join(labels_dir, label_file), train_labels_dir)

    for image_file, label_file in val_files:
        shutil.copy(os.path.join(images_dir, image_file), val_images_dir)
        shutil.copy(os.path.join(labels_dir, label_file), val_labels_dir)

    print(f"Split {len(matched_files)} files into {len(train_files)} train and {len(val_files)} validation files.")

images_dir = './dataset/images'  
labels_dir = './dataset/labels'  
output_dir = './dataset/yolo2'   

split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.8)

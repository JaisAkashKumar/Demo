import os
import random
import shutil

def split_dataset(source_dir, train_images_dir, train_labels_dir, val_images_dir, val_labels_dir, split_ratio=0.8, max_samples_per_class=500):
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)

    for species_folder in os.listdir(source_dir):
        species_path = os.path.join(source_dir, species_folder)

        if os.path.isdir(species_path):
            image_files = [f for f in os.listdir(species_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) > max_samples_per_class:
                print(f'Folder "{species_folder}" has more than 500 images.')
                image_files = random.sample(image_files, max_samples_per_class)

            random.shuffle(image_files)

            split_index = int(len(image_files) * split_ratio)

            train_images = image_files[:split_index]
            val_images = image_files[split_index:]

            def copy_files(image_list, image_dest_dir, label_dest_dir):
                for image in image_list:
                    image_path = os.path.join(species_path, image)
                    annotation_path = image_path.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')

                    sub_dir = os.path.join(image_dest_dir, species_folder)
                    os.makedirs(sub_dir, exist_ok=True)
                    shutil.copy(image_path, os.path.join(sub_dir, image))

                    if os.path.exists(annotation_path):
                        label_sub_dir = os.path.join(label_dest_dir, species_folder)
                        os.makedirs(label_sub_dir, exist_ok=True)
                        shutil.copy(annotation_path, os.path.join(label_sub_dir, os.path.basename(annotation_path)))

            copy_files(train_images, train_images_dir, train_labels_dir)
            copy_files(val_images, val_images_dir, val_labels_dir)

    print("Dataset split completed with a maximum of 500 images per class!")

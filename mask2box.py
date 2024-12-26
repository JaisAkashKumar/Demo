import os
import cv2

class_mapping = {f"fish_{i+1:02}": i for i in range(23)} 

def create_annotations_with_masks(images_directory, masks_directory):
    for species in os.listdir(images_directory):
        species_path = os.path.join(images_directory, species)
        masks_path = os.path.join(masks_directory, f"mask_{species.split('_')[-1]}")  

        if not os.path.exists(masks_path):
            print(f"Mask folder {masks_path} does not exist. Skipping...")
            continue

        for img_name in os.listdir(species_path):
            if img_name.endswith(('.png')):
                img_path = os.path.join(species_path, img_name)
                mask_path = os.path.join(masks_path, img_name.replace('fish', 'mask'));
        
                img = cv2.imread(img_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 

                if mask is None:
                    print(f"Mask not found for {img_name}. Skipping...")
                    continue

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if not contours:
                    print(f"No contours found in mask for {img_name}. Skipping...")
                    continue

                annotation_file = img_name.replace('.png', '.txt')
                annotation_path = os.path.join(species_path, annotation_file)

                with open(annotation_path, 'w') as f:
                    for contour in contours:
                        x, y, w, h = cv2.boundingRect(contour)
                        img_height, img_width = img.shape[:2]
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        norm_width = w / img_width
                        norm_height = h / img_height

                        class_idx = class_mapping[species]  
                        f.write(f"{class_idx} {x_center} {y_center} {norm_width} {norm_height}\n")

                print(f"Annotations saved to {annotation_path}")

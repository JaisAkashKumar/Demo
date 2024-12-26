import cv2
import os
import xml.etree.ElementTree as ET


def extract_frames(video_path, output_folder, video_id, xml_file, img_width, img_height, class_map):
    # video_output_folder = os.path.join(output_folder, video_id)
    images_folder = os.path.join(output_folder, 'images')
    labels_folder = os.path.join(output_folder, 'labels')

    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_filename = os.path.join(images_folder, f"{video_id}_frame{frame_count}.jpeg")
        cv2.imwrite(frame_filename, frame)

        yolo_annotations = process_single_frame_annotation(xml_file, frame_count, img_width, img_height, class_map)
        save_yolo_annotations(labels_folder, video_id, frame_count, yolo_annotations)
        
        frame_count += 1

    cap.release()

def process_single_frame_annotation(xml_file, frame_count, img_width, img_height, class_map):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        yolo_annotations = []
        frame_found = False

        for frame in root.findall('frame'):
            frame_id = int(frame.get('id'))
            if frame_id == frame_count:
                frame_found = True
                
                for obj in frame.findall('object'):
                    class_name = obj.get('fish_species').strip().lower()  
                    if class_name in class_map:
                        class_id = class_map[class_name]
                        x = float(obj.get('x'))
                        y = float(obj.get('y'))
                        w = float(obj.get('w'))
                        h = float(obj.get('h'))

                        # Calculate YOLO format values
                        x_center = (x + w / 2) / img_width
                        y_center = (y + h / 2) / img_height
                        width = w / img_width
                        height = h / img_height
                        
                        annotation = f"{class_id} {x_center} {y_center} {width} {height}"
                        yolo_annotations.append(annotation)
                
        if not frame_found:
            print(f"Frame ID {frame_count} not found in {xml_file}")

        return yolo_annotations

def save_yolo_annotations(output_dir, video_id, frame_count, yolo_annotations):
    yolo_file_path = os.path.join(output_dir, f"{video_id}_frame{frame_count}.txt")
    if yolo_annotations:
        with open(yolo_file_path, 'w') as f:
            for line in yolo_annotations:
                f.write(line + '\n')
    # else:
    #     with open(yolo_file_path, 'w') as f:
    #         f.write("0\n") 


species_names = [
"Abudefduf Vaigiensis",
"Acanthurus Nigrofuscus",
"Amphiprion Clarkii",
"Chaetodon Lunulatus",
"Chaetodon Speculum",
"Chaetodon Trifascialis",
"Chromis Chrysura",
"Dascyllus Aruanus",
"Dascyllus Reticulatus",
"Hemigymnus Melapterus",
"Myripristis Kuntee",
"Neoglyphidodon Nigroris",
"Pempheris Vanicolensis",
"Plectrogly-Phidodon Dickii" ,
"Zebrasoma Scopas",
"No Fish Present"
]

class_map = {name.strip().lower(): idx for idx, name in enumerate(species_names)}

input_video_folder = './fishclef_2015_release/training_set/videos' 
input_xml_folder = './fishclef_2015_release/training_set/gt'
output_folder = './dataset'


for video_file in os.listdir(input_video_folder):
    if video_file.endswith('.flv'):
        video_id = os.path.splitext(video_file)[0] 
        video_path = os.path.join(input_video_folder, video_file)
        xml_file = os.path.join(input_xml_folder, f"{video_id}.xml") 

        cap = cv2.VideoCapture(video_path)
        img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        video_output_folder = os.path.join(output_folder, video_id)
        extract_frames(video_path, output_folder, video_id, xml_file, img_width, img_height, class_map)


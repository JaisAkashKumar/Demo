import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf



bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

def preprocess_for_model(fish_crop):
    resized_crop = cv2.resize(fish_crop, (200, 200))  
    normalized_crop = resized_crop / 255.0
    fish_input = np.expand_dims(normalized_crop, axis=0) 
    return fish_input

def create_rgb_blob_image(gmm_mask, optical_flow_mask):
    # Create RGB image with GMM mask in green and optical flow mask in red
    rgb_blob_image = np.zeros((gmm_mask.shape[0], gmm_mask.shape[1], 3), dtype=np.uint8)
    rgb_blob_image[:, :, 1] = gmm_mask  # Green channel for GMM
    rgb_blob_image[:, :, 0] = optical_flow_mask  # Red channel for optical flow
    return rgb_blob_image

def process_frame(frame, prev_frame=None):
    # GMM background subtraction
    gmm_fg_mask = bg_subtractor.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gmm_fg_mask = cv2.morphologyEx(gmm_fg_mask, cv2.MORPH_OPEN, kernel)
    
    # Optical flow if previous frame is available
    optical_flow_fg_mask = np.zeros_like(gmm_fg_mask)
    if prev_frame is not None:
        optical_flow_fg_mask = bg_subtractor.apply(frame)
        optical_flow_fg_mask = cv2.morphologyEx(optical_flow_fg_mask, cv2.MORPH_OPEN, kernel)

    # Merge GMM and optical flow masks into an RGB image
    rgb_blob_image = create_rgb_blob_image(gmm_fg_mask, optical_flow_fg_mask)
    
    # Extract contours from the combined mask
    contours, _ = cv2.findContours(cv2.bitwise_or(gmm_fg_mask, optical_flow_fg_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        if 300 < cv2.contourArea(contour) < 5000:  # Filter by size
            x, y, w, h = cv2.boundingRect(contour)
            fish_crop = frame[y:y + h, x:x + w]  # Crop fish region

            fish_input = preprocess_for_model(fish_crop)
            predictions = model.predict(fish_input)
            confidence = np.max(predictions)
            species_label = np.argmax(predictions)

            species_names = ["Abudefduf Vaigiensis", "Acanthurus Nigrofuscus", "Amphiprion Clarkii", "Chaetodon Lunulatus", "Chaetodon Speculum", "Chaetodon Trifascialis", "Chromis Chrysura", "Dascyllus Aruanus" , "Dascyllus Reticulatus", "Hemigymnus Melapterus", "Myripristis Kuntee", "Neoglyphidodon Nigroris", "Pempheris Vanicolensis", "Plectrogly-Phidodon Dickii", "Zebrasoma scopas"]
            species_name = species_names[species_label] if species_label < len(species_names) else "Unknown"

            detections.append((x, y, w, h, confidence, species_name))

    return frame, rgb_blob_image, detections

def adjust_label_position(x, y, w, h, label_size, frame_width, frame_height):
    label_x, label_y = x, y - 10
    if label_x + label_size[0] > frame_width:
        label_x = x + w - label_size[0]
    if label_x < 0:
        label_x = x
    if label_y - label_size[1] < 0:
        label_y = y + h + label_size[1]
    return label_x, label_y

def show_frame(frame):
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

def main():
    cap = cv2.VideoCapture('./fishclef_2015_release/test_set/videos/sub_0a3548f4df96ac98d7f226aa1125fd06#201103090650_0.flv')

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('./fishclef_2015_release/test_set/output_video_merged2.mp4', fourcc, fps, (width, height))

    ret, prev_frame = cap.read()

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        processed_frame, rgb_blob_image, detections = process_frame(current_frame, prev_frame)

        for (x, y, w, h, confidence, species_name) in detections:
            cv2.rectangle(processed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label = f"{species_name} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
            label_x, label_y = adjust_label_position(x, y, w, h, label_size, processed_frame.shape[1], processed_frame.shape[0])
            cv2.putText(processed_frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        out.write(processed_frame)
        show_frame(processed_frame)
        time.sleep(1 / fps)
        
        prev_frame = current_frame

    cap.release()
    out.release()
    print("Done processing video with labeled confidence scores and species names.")

if __name__ == "__main__":
    main()
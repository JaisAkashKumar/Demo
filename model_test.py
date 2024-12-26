import os
import cv2
from ultralytics import YOLO 

# Load the YOLO model
model_path = './runs/detect/train/weights/best.pt'
model = YOLO(model_path)  

def predict_and_draw_boxes(input_path):
    # Check if the input is a video or an image
    if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.flv')):
        # Process video
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can also try 'mp4v' for .mp4 format
        output_path = os.path.join('output', os.path.splitext(os.path.basename(input_path))[0] + '.avi')

        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = model(frame)

            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])  
                conf = result.conf[0]  
                cls = int(result.cls[0])
                label = f'{model.names[cls]} {conf:.2f}'  

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Write the processed frame to the output video
            out.write(frame)


        cap.release()
        out.release()
        print(f'Saved predictions for video: {input_path} to {output_path}')
    else:
        # Process image
        img = cv2.imread(input_path)

        # Perform inference
        results = model(img)

        for result in results[0].boxes:
            x1, y1, x2, y2 = map(int, result.xyxy[0])  
            conf = result.conf[0]  
            cls = int(result.cls[0])
            label = f'{model.names[cls]} {conf:.2f}'  

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        output_image_path = os.path.join('output', os.path.basename(input_path))
        os.makedirs('output', exist_ok=True)
        cv2.imwrite(output_image_path, img)

        print(f'Saved prediction for image: {input_path} to {output_image_path}')


video_folder = './fishclef_2015_release/test_set/videos/';
input_files = []

input_files.append(os.path.join(video_folder, 'sub_6d9cde8d43ceff4eecc9fd2259d55063#201103120620_5.flv'))

# for file_name in os.listdir(video_folder): 
#     input_files.append(os.path.join(video_folder, file_name))

for file_path in input_files:
    predict_and_draw_boxes(file_path)

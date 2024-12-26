# First PreProcess the image from mask to obtain the bounding box for each image 
# Now Split the dataset in 80:20 ratio for train and validation
# Now train on YOLO 
# Now train on GMM and Optical Flow
# Now train CNN - SENET for classification 
# Now using these 2 models YOLO + Gmm & optical flow + SENET for classification build a pipeline that merges the result of both and incase of intersection prefers yolo classification


# def preprocess_with_gmm_optical_flow(video_path):
#     # Load the video
#     cap = cv2.VideoCapture(video_path)33weewr
    
#     # GMM for background subtraction
#     backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
#     ret, prev_frame = cap.read()
#     prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Apply GMM to detect motion
#         fgMask = backSub.apply(frame)
        
#         # Optical Flow (Dense)
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#         prev_gray = gray
        
#         # Visualize motion (optional)
#         hsv = np.zeros_like(frame)
#         hsv[..., 1] = 255
#         mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#         hsv[..., 0] = ang * 180 / np.pi / 2
#         hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#         motion_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
#         # Extract bounding boxes from motion (you'll need to refine this for detection)
#         contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for contour in contours:
#             if cv2.contourArea(contour) > 500:
#                 x, y, w, h = cv2.boundingRect(contour)
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Show result
#         cv2.imshow('Frame', frame)
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



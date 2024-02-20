import os
import numpy as np
import cv2
import mediapipe as mp
from keras_facenet import FaceNet
import time


frame_count = 0
start_time = time.time()
embedder = FaceNet()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def get_face_embeddings(face_roi):
    if face_roi is not None and face_roi.size > 0:
        resized_face = cv2.resize(face_roi, (160, 160))
        if resized_face.size > 0:
            image = np.expand_dims(resized_face, axis=0)
            embeddings = embedder.embeddings(image)
            return embeddings
    return None


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            images.append(image)
            labels.append(folder.split('/')[-1])  # Assuming folder structure is 'folder_name/image.jpg'
    return images, labels


known_faces = []
known_labels = []
folders_path = "ps"


for folder_name in os.listdir(folders_path):
    folder_path = os.path.join(folders_path, folder_name)
    if os.path.isdir(folder_path):
        images, labels = load_images_from_folder(folder_path)
        known_faces.extend(images)
        known_labels.extend(labels)


known_embeddings = []
for face_image in known_faces:
    rgb_frame = face_image[:, :, ::-1]
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            # Get the bounding box coordinates
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = face_image.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
           


confidence_threshold = 0.7

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    results = face_detection.process(rgb_frame)
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                   int(bboxC.width * iw), int(bboxC.height * ih)
            face_roi = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            if face_roi is not None and face_roi.size > 0:
                frame_embeddings = get_face_embeddings(face_roi)
                if frame_embeddings is not None:  
                    for i, frame_embedding in enumerate(frame_embeddings):
                        distances = []
                        for known_emb in known_embeddings:
                            dist = np.linalg.norm(known_emb - frame_embedding, axis=1)
                            distances.append(np.mean(dist))

                        min_distance = np.min(distances)
                        recognized_label = known_labels[np.argmin(distances)] if min_distance < confidence_threshold else "Unknown"
                        
                        # Display the recognized label and distance on the frame for each face
                        cv2.putText(frame, recognized_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

    # Add timestamp to the frame
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Frames", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

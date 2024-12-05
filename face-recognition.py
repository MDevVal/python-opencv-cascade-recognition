import cv2
import os
import numpy as np

cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


def load_training_data(faces_dir="faces"):
    images = []
    labels = []
    label_names = {}
    current_label = 0

    for person_name in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        label_names[current_label] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in faces:
                face_roi = img[y : y + h, x : x + w]
                images.append(face_roi)
                labels.append(current_label)
                break
        current_label += 1

    return images, labels, label_names


def train_recognizer(images, labels):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer


if __name__ == "__main__":
    images, labels, label_names = load_training_data("faces")

    if len(images) == 0:
        print("No training images found. Check the faces directory.")
        exit(1)

    recognizer = train_recognizer(images, labels)
    threshold = 70

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            face_roi = gray[y : y + h, x : x + w]
            label, confidence = recognizer.predict(face_roi)
            if confidence < threshold:
                person_name = label_names.get(label, "Unknown")
                print(f"Recognized as {person_name} with confidence {confidence}")
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{person_name} ({confidence:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(
                    frame,
                    "Unknown",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

    cap.release()
    cv2.destroyAllWindows()

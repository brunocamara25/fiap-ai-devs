import cv2
import face_recognition
import os
import numpy as np
from ingestion import VideoIngestor
from face_detection import FaceDetector
from activity_recog import ActivityRecognizer
from anomaly_detection import AnomalyDetector
from summarizer import Summarizer
from keypoints_extractor import extract_keypoints

def analyze_scene(video_path, summarizer, face_detector, activity_recognizer, anomaly_detector):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Reconhecimento facial
        locations, names, attributes = face_detector.detect(frame)
        emotions = attributes if names else [{"dominant_emotion": "Desconhecido"}]

        # 2. Atividades
        activity = activity_recognizer.recognize(frame)

        # 3. Anomalias
        anomaly = anomaly_detector.detect(frame)

        # 4. Desenhar marcações
        for (top, right, bottom, left), name in zip(locations, names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            text = f"{name}: {emotions[0]['dominant_emotion']}, {activity}"
            cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        anomaly_text = f"Anomaly: {anomaly}"
        cv2.putText(frame, anomaly_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if anomaly else (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_number}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # 5. Exibir o frame
        cv2.imshow("Video Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 6. Atualiza resumo
        summarizer.update(
            emotions[0]['dominant_emotion'] if emotions else "Desconhecido",
            activity,
            anomaly,
            names=names
        )

        frame_number += 1
    cap.release()
    cv2.destroyAllWindows()


def main():
    video_path = "../data/video.mp4"

    face_detector = FaceDetector()
    activity_recognizer = ActivityRecognizer()
    anomaly_detector = AnomalyDetector()
    summarizer = Summarizer()

    analyze_scene(video_path, summarizer, face_detector, activity_recognizer, anomaly_detector)

    summarizer.save("../reports/summary.json")

if __name__ == "__main__":
    main()

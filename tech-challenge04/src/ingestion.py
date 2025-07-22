import cv2

class VideoIngestor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError("Erro ao abrir o vídeo.")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame

    def release(self):
        self.cap.release()
import cv2
import numpy as np

def detect_cuts(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    frame_number = 0
    cuts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Converte para escala de cinza para reduzir a complexidade
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Se é o primeiro frame ou não há um frame anterior, armazena e continua
        if prev_frame is None:
            prev_frame = gray_frame
            frame_number += 1
            continue

        # Calcula a diferença entre os frames consecutivos
        diff = cv2.absdiff(prev_frame, gray_frame)
        non_zero_count = np.count_nonzero(diff)

        # Defina um limite para considerar uma mudança significativa (corte)
        if non_zero_count > 50000:  # Esse valor pode ser ajustado conforme necessário
            cuts.append(frame_number)

        # Atualiza o frame anterior
        prev_frame = gray_frame
        frame_number += 1

    cap.release()
    return cuts

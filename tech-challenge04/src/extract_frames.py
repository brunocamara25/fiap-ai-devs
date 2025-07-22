import cv2

def extract_frames(video_path, cuts):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    extracted_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Se o frame atual for um corte, armazene a imagem
        if frame_number in cuts:
            frame_filename = f"frame_{frame_number}.jpg"
            cv2.imwrite(frame_filename, frame)
            extracted_frames.append(frame_filename)

        frame_number += 1

    cap.release()
    return extracted_frames

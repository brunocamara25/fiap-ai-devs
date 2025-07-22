
import cv2
import mediapipe as mp
import numpy as np

class ActivityRecognizer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=False)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False)
        self.prev_landmarks = None
        self.movement_history = []
        self.mouth_open_history = []
        self.mouth_width_history = []

    def recognize(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        face_results = self.face_mesh.process(rgb)
        movement = self._recognize_movement(results, frame.shape)
        reaction = self._recognize_reaction(face_results)
        return {"movement": movement, "reaction": reaction}

    def _recognize_movement(self, results, frame_shape):
        if not results.pose_landmarks:
            return "No Pose"

        height, width, _ = frame_shape
        mp_pose = self.mp_pose
        landmarks = np.array([(lm.x * width, lm.y * height, lm.z * width) for lm in results.pose_landmarks.landmark])

        def dist(a, b):
            return np.linalg.norm(landmarks[a] - landmarks[b])

        # Distâncias relevantes
        shoulder_dist = dist(mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        hip_dist = dist(mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value)
        shoulder_to_hip = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value][1]) / 2 -                           (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value][1] + landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value][1]) / 2

        # Classificação simplificada baseada nas proporções
        if shoulder_to_hip < 10:
            return "Lying Down"
        elif hip_dist > shoulder_dist * 1.2:
            return "Sitting"
        elif hip_dist < shoulder_dist * 0.8:
            return "Standing"

        # Detecção de movimento se posição não for clara
        if self.prev_landmarks is not None:
            movement = np.linalg.norm(landmarks - self.prev_landmarks, axis=1).mean()
            self.movement_history.append(movement)
            if len(self.movement_history) > 5:
                self.movement_history.pop(0)
            avg_movement = np.mean(self.movement_history)
            if avg_movement > 15:
                return "Running"
            elif avg_movement > 7:
                return "Walking"
            elif avg_movement > 3:
                return "Dancing"
            else:
                return "Standing"

        self.prev_landmarks = landmarks
        return "Standing"

    def _recognize_reaction(self, face_results):
        if not face_results.multi_face_landmarks:
            return None
        face_landmarks = face_results.multi_face_landmarks[0]
        upper_lip = face_landmarks.landmark[13]
        lower_lip = face_landmarks.landmark[14]
        mouth_open = abs(upper_lip.y - lower_lip.y)
        self.mouth_open_history.append(mouth_open)
        if len(self.mouth_open_history) > 5:
            self.mouth_open_history.pop(0)
        avg_mouth_open = np.mean(self.mouth_open_history)

        left_mouth = face_landmarks.landmark[61]
        right_mouth = face_landmarks.landmark[291]
        mouth_width = abs(left_mouth.x - right_mouth.x)
        self.mouth_width_history.append(mouth_width)
        if len(self.mouth_width_history) > 5:
            self.mouth_width_history.pop(0)
        avg_mouth_width = np.mean(self.mouth_width_history)

        nose_tip = face_landmarks.landmark[1]
        chin = face_landmarks.landmark[152]
        head_down = nose_tip.y > chin.y - 0.02

        if avg_mouth_open > 0.08 and avg_mouth_width < 0.08:
            return "Surprised"
        elif avg_mouth_width > 0.08 and avg_mouth_open > 0.04:
            return "Laughing"
        elif avg_mouth_width > 0.08:
            return "Smiling"
        elif avg_mouth_open > 0.03:
            return "Talking"
        elif head_down and avg_mouth_open < 0.03:
            return "Reading"
        return "Neutral"

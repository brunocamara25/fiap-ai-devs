#!/usr/bin/env python3
"""
Face Detector - Módulo para detecção de rostos usando MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional

class FaceDetector:
    """
    Detecta rostos em imagens usando MediaPipe Face Detection.
    
    Por que MediaPipe?
    - Rápido e eficiente (roda em tempo real)
    - Boa precisão mesmo com rostos pequenos
    - Detecta múltiplos rostos simultaneamente
    - Fornece pontos de referência faciais
    """
    
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        """
        Inicializa o detector de rostos.
        
        Args:
            min_detection_confidence (float): Confiança mínima para detecção (0-1)
                - 0.5 = padrão, bom equilíbrio
                - 0.7 = mais preciso, mas pode perder alguns rostos
                - 0.3 = detecta mais, mas pode ter falsos positivos
            
            model_selection (int): Modelo a usar
                - 0 = modelo para rostos próximos (< 2 metros)
                - 1 = modelo para rostos distantes (< 5 metros)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando FaceDetector")
        
        # Inicializar MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
        # Contador para IDs únicos de rostos
        self.face_id_counter = 0
        
        # Cache para rastreamento básico
        self.previous_faces = []
        
    def detect_faces(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """
        Detecta rostos em um frame.
        
        Args:
            frame (np.ndarray): Imagem BGR do OpenCV
            frame_number (int): Número do frame para referência
            
        Returns:
            List[Dict]: Lista de rostos detectados com suas propriedades
        """
        # Converter BGR para RGB (MediaPipe usa RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Executar detecção
        results = self.face_detection.process(rgb_frame)
        
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            
            for idx, detection in enumerate(results.detections):
                # Extrair bounding box
                bbox = detection.location_data.relative_bounding_box
                
                # Converter coordenadas relativas (0-1) para absolutas (pixels)
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Garantir que as coordenadas estão dentro da imagem
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Extrair pontos de referência (olhos, nariz, boca, orelhas)
                landmarks = self._extract_landmarks(detection, w, h)
                
                # Criar dicionário com informações do rosto
                face_data = {
                    'face_id': f"face_{frame_number}_{idx}",
                    'frame_number': frame_number,
                    'timestamp': frame_number / 30.0,  # Assumindo 30 FPS
                    'bbox': {
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    },
                    'confidence': detection.score[0],
                    'landmarks': landmarks,
                    'center': (x + width // 2, y + height // 2),
                    'area': width * height
                }
                
                faces.append(face_data)
                
        self.logger.debug(f"Frame {frame_number}: {len(faces)} rostos detectados")
        return faces
    
    def _extract_landmarks(self, detection, img_width: int, img_height: int) -> Dict:
        """
        Extrai pontos de referência facial.
        
        MediaPipe fornece 6 pontos:
        - 0: Olho direito
        - 1: Olho esquerdo  
        - 2: Ponta do nariz
        - 3: Centro da boca
        - 4: Orelha direita
        - 5: Orelha esquerda
        """
        landmarks = {}
        
        if hasattr(detection, 'location_data') and detection.location_data.relative_keypoints:
            keypoints = detection.location_data.relative_keypoints
            landmark_names = ['right_eye', 'left_eye', 'nose_tip', 'mouth_center', 
                            'right_ear', 'left_ear']
            
            for i, (keypoint, name) in enumerate(zip(keypoints, landmark_names)):
                landmarks[name] = {
                    'x': int(keypoint.x * img_width),
                    'y': int(keypoint.y * img_height)
                }
        
        return landmarks
    
    def draw_faces(self, frame: np.ndarray, faces: List[Dict], 
                   draw_landmarks: bool = True) -> np.ndarray:
        """
        Desenha os rostos detectados no frame.
        
        Args:
            frame: Imagem onde desenhar
            faces: Lista de rostos detectados
            draw_landmarks: Se deve desenhar pontos de referência
            
        Returns:
            Frame com anotações
        """
        annotated_frame = frame.copy()
        
        for face in faces:
            bbox = face['bbox']
            confidence = face['confidence']
            
            # Cor baseada na confiança (verde = alta, amarelo = média, vermelho = baixa)
            if confidence > 0.8:
                color = (0, 255, 0)  # Verde
            elif confidence > 0.6:
                color = (0, 255, 255)  # Amarelo
            else:
                color = (0, 0, 255)  # Vermelho
            
            # Desenhar bounding box
            cv2.rectangle(
                annotated_frame,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color, 2
            )
            
            # Adicionar texto com ID e confiança
            text = f"{face['face_id']}: {confidence:.2f}"
            cv2.putText(
                annotated_frame,
                text,
                (bbox['x'], bbox['y'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Desenhar landmarks se solicitado
            if draw_landmarks and face['landmarks']:
                for landmark_name, point in face['landmarks'].items():
                    cv2.circle(
                        annotated_frame,
                        (point['x'], point['y']),
                        3,
                        (255, 0, 255),  # Magenta
                        -1
                    )
        
        return annotated_frame
    
    def extract_face_roi(self, frame: np.ndarray, face: Dict, 
                        padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extrai a região de interesse (ROI) do rosto.
        
        Args:
            frame: Imagem completa
            face: Dados do rosto detectado
            padding: Percentual de padding ao redor do rosto (0.2 = 20%)
            
        Returns:
            Imagem recortada do rosto ou None se houver erro
        """
        try:
            bbox = face['bbox']
            h, w = frame.shape[:2]
            
            # Adicionar padding
            pad_w = int(bbox['width'] * padding)
            pad_h = int(bbox['height'] * padding)
            
            # Calcular nova região com padding
            x1 = max(0, bbox['x'] - pad_w)
            y1 = max(0, bbox['y'] - pad_h)
            x2 = min(w, bbox['x'] + bbox['width'] + pad_w)
            y2 = min(h, bbox['y'] + bbox['height'] + pad_h)
            
            # Extrair ROI
            face_roi = frame[y1:y2, x1:x2]
            
            # Verificar se a ROI é válida
            if face_roi.size == 0:
                return None
                
            return face_roi
            
        except Exception as e:
            self.logger.error(f"Erro ao extrair ROI do rosto: {str(e)}")
            return None
    
    def estimate_face_quality(self, face: Dict, frame: np.ndarray) -> float:
        """
        Estima a qualidade da detecção do rosto.
        
        Fatores considerados:
        - Tamanho do rosto (muito pequeno = baixa qualidade)
        - Confiança da detecção
        - Posição (rostos nas bordas podem estar cortados)
        
        Returns:
            float: Score de qualidade (0-1)
        """
        bbox = face['bbox']
        h, w = frame.shape[:2]
        
        # Fator 1: Tamanho relativo do rosto
        face_area = bbox['width'] * bbox['height']
        frame_area = w * h
        size_ratio = face_area / frame_area
        size_score = min(1.0, size_ratio * 20)  # Normalizar
        
        # Fator 2: Confiança da detecção
        confidence_score = face['confidence']
        
        # Fator 3: Distância das bordas
        margin = 20  # pixels
        x_margin = min(bbox['x'], w - (bbox['x'] + bbox['width']))
        y_margin = min(bbox['y'], h - (bbox['y'] + bbox['height']))
        margin_score = min(1.0, min(x_margin, y_margin) / margin)
        
        # Combinar scores (média ponderada)
        quality = (size_score * 0.3 + confidence_score * 0.5 + margin_score * 0.2)
        
        return quality
    
    def cleanup(self):
        """Libera recursos do MediaPipe."""
        if self.face_detection:
            self.face_detection.close()
            

# Função de teste
def test_face_detector():
    """Testa o detector de rostos com uma imagem ou vídeo."""
    import matplotlib.pyplot as plt
    
    # Inicializar detector
    detector = FaceDetector(min_detection_confidence=0.5)
    
    # Testar com webcam ou vídeo
    cap = cv2.VideoCapture(0)  # Use 0 para webcam ou caminho para vídeo
    
    print("Pressione 'q' para sair, 's' para salvar frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar rostos
        faces = detector.detect_faces(frame, 0)
        
        # Desenhar resultados
        annotated = detector.draw_faces(frame, faces)
        
        # Mostrar estatísticas
        cv2.putText(
            annotated,
            f"Rostos: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Mostrar frame
        cv2.imshow('Face Detection', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite('face_detection_test.jpg', annotated)
            print("Frame salvo!")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()


if __name__ == "__main__":
    test_face_detector()
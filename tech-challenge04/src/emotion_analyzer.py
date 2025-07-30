#!/usr/bin/env python3
"""
Emotion Analyzer - Módulo para análise de emoções faciais usando DeepFace
"""

import cv2
import numpy as np
from deepface import DeepFace
import logging
from typing import List, Dict, Optional, Tuple
import warnings

# Suprimir warnings do TensorFlow
warnings.filterwarnings('ignore')

class EmotionAnalyzer:
    """
    Analisa emoções em rostos detectados usando DeepFace.
    
    DeepFace pode detectar 7 emoções:
    - angry (raiva)
    - disgust (nojo)
    - fear (medo)
    - happy (feliz)
    - sad (triste)
    - surprise (surpresa)
    - neutral (neutro)
    """
    
    # Tradução das emoções para português
    EMOTION_TRANSLATIONS = {
        'angry': 'raiva',
        'disgust': 'nojo',
        'fear': 'medo',
        'happy': 'feliz',
        'sad': 'triste',
        'surprise': 'surpresa',
        'neutral': 'neutro'
    }
    
    # Cores para visualização (BGR)
    EMOTION_COLORS = {
        'angry': (0, 0, 255),      # Vermelho
        'disgust': (0, 128, 128),  # Verde escuro
        'fear': (255, 0, 255),     # Magenta
        'happy': (0, 255, 0),      # Verde
        'sad': (255, 0, 0),        # Azul
        'surprise': (0, 255, 255), # Amarelo
        'neutral': (128, 128, 128) # Cinza
    }
    
    def __init__(self, model_name: str = 'VGG-Face', 
                 detector_backend: str = 'opencv',
                 enforce_detection: bool = False):
        """
        Inicializa o analisador de emoções.
        
        Args:
            model_name: Modelo para análise ('VGG-Face', 'Facenet', 'OpenFace', etc)
            detector_backend: Backend para detecção ('opencv', 'ssd', 'mtcnn', etc)
            enforce_detection: Se True, gera erro quando não detecta rosto
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Inicializando EmotionAnalyzer com modelo {model_name}")
        
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        
        # Cache para otimização
        self.emotion_cache = {}
        
        # Estatísticas
        self.stats = {
            'total_analyzed': 0,
            'successful': 0,
            'failed': 0,
            'emotion_counts': {emotion: 0 for emotion in self.EMOTION_TRANSLATIONS.keys()}
        }
        
    def analyze_face_emotion(self, face_image: np.ndarray, 
                           face_id: str = None) -> Optional[Dict]:
        """
        Analisa a emoção em uma imagem de rosto.
        
        Args:
            face_image: Imagem do rosto (ROI)
            face_id: ID único do rosto para cache
            
        Returns:
            Dict com análise de emoções ou None se falhar
        """
        try:
            # Verificar cache
            if face_id and face_id in self.emotion_cache:
                return self.emotion_cache[face_id]
            
            # Garantir que a imagem tem tamanho mínimo
            if face_image.shape[0] < 48 or face_image.shape[1] < 48:
                self.logger.warning(f"Imagem muito pequena: {face_image.shape}")
                return None
            
            # Analisar com DeepFace
            result = DeepFace.analyze(
                img_path=face_image,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend
            )
            
            # DeepFace retorna lista, pegar primeiro resultado
            if isinstance(result, list):
                result = result[0]
            
            # Extrair dados de emoção
            emotion_data = {
                'dominant_emotion': result['dominant_emotion'],
                'emotion_scores': result['emotion'],
                'confidence': max(result['emotion'].values()) / 100.0,
                'face_id': face_id
            }
            
            # Adicionar tradução
            emotion_data['dominant_emotion_pt'] = self.EMOTION_TRANSLATIONS.get(
                result['dominant_emotion'], 
                result['dominant_emotion']
            )
            
            # Atualizar estatísticas
            self.stats['total_analyzed'] += 1
            self.stats['successful'] += 1
            self.stats['emotion_counts'][result['dominant_emotion']] += 1
            
            # Cachear resultado
            if face_id:
                self.emotion_cache[face_id] = emotion_data
            
            return emotion_data
            
        except Exception as e:
            self.logger.error(f"Erro ao analisar emoção: {str(e)}")
            self.stats['total_analyzed'] += 1
            self.stats['failed'] += 1
            return None
    
    def analyze_faces_emotions(self, frame: np.ndarray, 
                             faces: List[Dict]) -> List[Dict]:
        """
        Analisa emoções em múltiplos rostos de um frame.
        
        Args:
            frame: Frame completo do vídeo
            faces: Lista de rostos detectados (do FaceDetector)
            
        Returns:
            Lista de análises de emoção
        """
        emotions = []
        
        for face in faces:
            # Extrair região do rosto
            bbox = face['bbox']
            face_roi = frame[
                bbox['y']:bbox['y'] + bbox['height'],
                bbox['x']:bbox['x'] + bbox['width']
            ]
            
            # Analisar emoção
            emotion_result = self.analyze_face_emotion(
                face_roi, 
                face_id=face['face_id']
            )
            
            if emotion_result:
                # Combinar com dados do rosto
                emotion_data = {
                    **face,  # Incluir todos os dados do rosto
                    **emotion_result,  # Adicionar dados de emoção
                    'frame_number': face['frame_number']
                }
                emotions.append(emotion_data)
        
        return emotions
    
    def detect_emotion_anomalies(self, emotions: List[Dict], 
                               threshold: float = 0.85) -> List[Dict]:
        """
        Detecta anomalias emocionais (expressões exageradas).
        
        Uma anomalia emocional pode ser:
        - Emoção com intensidade muito alta (> threshold)
        - Mudança brusca de emoção
        - Combinação incomum de emoções
        
        Args:
            emotions: Lista de emoções analisadas
            threshold: Limiar para considerar emoção exagerada
            
        Returns:
            Lista de anomalias detectadas
        """
        anomalies = []
        
        for emotion in emotions:
            scores = emotion['emotion_scores']
            dominant = emotion['dominant_emotion']
            confidence = emotion['confidence']
            
            # Anomalia 1: Intensidade muito alta
            if confidence > threshold:
                anomaly = {
                    'type': 'high_intensity_emotion',
                    'description': f'Emoção {dominant} com intensidade {confidence:.2%}',
                    'face_id': emotion['face_id'],
                    'frame_number': emotion['frame_number'],
                    'emotion': dominant,
                    'intensity': confidence,
                    'severity': 'high' if confidence > 0.9 else 'medium'
                }
                anomalies.append(anomaly)
            
            # Anomalia 2: Múltiplas emoções fortes
            strong_emotions = [e for e, score in scores.items() if score > 70]
            if len(strong_emotions) > 2:
                anomaly = {
                    'type': 'mixed_strong_emotions',
                    'description': f'Múltiplas emoções fortes: {", ".join(strong_emotions)}',
                    'face_id': emotion['face_id'],
                    'frame_number': emotion['frame_number'],
                    'emotions': strong_emotions,
                    'severity': 'medium'
                }
                anomalies.append(anomaly)
            
            # Anomalia 3: Emoções específicas como anomalias
            if dominant in ['fear', 'disgust', 'surprise'] and confidence > 0.7:
                anomaly = {
                    'type': 'unusual_expression',
                    'description': f'Expressão incomum: {dominant} ({confidence:.2%})',
                    'face_id': emotion['face_id'],
                    'frame_number': emotion['frame_number'],
                    'emotion': dominant,
                    'intensity': confidence,
                    'severity': 'low'
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def visualize_emotions(self, frame: np.ndarray, 
                         emotions: List[Dict]) -> np.ndarray:
        """
        Visualiza as emoções detectadas no frame.
        
        Args:
            frame: Frame original
            emotions: Lista de emoções detectadas
            
        Returns:
            Frame anotado com visualização de emoções
        """
        annotated = frame.copy()
        
        for emotion in emotions:
            bbox = emotion['bbox']
            dominant = emotion['dominant_emotion']
            confidence = emotion['confidence']
            
            # Cor baseada na emoção
            color = self.EMOTION_COLORS.get(dominant, (255, 255, 255))
            
            # Desenhar barra de emoção acima do rosto
            bar_height = 20
            bar_y = max(0, bbox['y'] - bar_height - 5)
            
            # Barra de fundo
            cv2.rectangle(
                annotated,
                (bbox['x'], bar_y),
                (bbox['x'] + bbox['width'], bar_y + bar_height),
                (0, 0, 0),
                -1
            )
            
            # Barra de progresso
            bar_width = int(bbox['width'] * confidence)
            cv2.rectangle(
                annotated,
                (bbox['x'], bar_y),
                (bbox['x'] + bar_width, bar_y + bar_height),
                color,
                -1
            )
            
            # Texto com emoção
            text = f"{self.EMOTION_TRANSLATIONS[dominant]}: {confidence:.1%}"
            cv2.putText(
                annotated,
                text,
                (bbox['x'], bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # Mini gráfico de todas as emoções
            self._draw_emotion_chart(annotated, emotion, bbox)
        
        return annotated
    
    def _draw_emotion_chart(self, frame: np.ndarray, emotion: Dict, bbox: Dict):
        """
        Desenha um mini gráfico com todas as emoções.
        """
        scores = emotion['emotion_scores']
        chart_x = bbox['x'] + bbox['width'] + 10
        chart_y = bbox['y']
        bar_width = 100
        bar_height = 10
        spacing = 2
        
        # Verificar se há espaço para desenhar
        if chart_x + bar_width > frame.shape[1]:
            return
        
        for i, (emo, score) in enumerate(scores.items()):
            y = chart_y + i * (bar_height + spacing)
            
            # Barra de fundo
            cv2.rectangle(
                frame,
                (chart_x, y),
                (chart_x + bar_width, y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Barra de progresso
            progress_width = int(bar_width * (score / 100))
            color = self.EMOTION_COLORS.get(emo, (255, 255, 255))
            cv2.rectangle(
                frame,
                (chart_x, y),
                (chart_x + progress_width, y + bar_height),
                color,
                -1
            )
            
            # Label
            label = f"{emo[:3]}: {score:.0f}%"
            cv2.putText(
                frame,
                label,
                (chart_x + bar_width + 5, y + bar_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (255, 255, 255),
                1
            )
    
    def get_emotion_summary(self) -> Dict:
        """
        Retorna um resumo das análises de emoção.
        """
        total = self.stats['total_analyzed']
        if total == 0:
            return {'message': 'Nenhuma análise realizada ainda'}
        
        summary = {
            'total_analyzed': total,
            'successful': self.stats['successful'],
            'failed': self.stats['failed'],
            'success_rate': self.stats['successful'] / total,
            'emotion_distribution': {}
        }
        
        # Calcular distribuição percentual
        for emotion, count in self.stats['emotion_counts'].items():
            percentage = (count / self.stats['successful'] * 100) if self.stats['successful'] > 0 else 0
            summary['emotion_distribution'][emotion] = {
                'count': count,
                'percentage': round(percentage, 2),
                'translation': self.EMOTION_TRANSLATIONS[emotion]
            }
        
        # Emoção mais comum
        if self.stats['emotion_counts']:
            most_common = max(self.stats['emotion_counts'].items(), key=lambda x: x[1])
            summary['most_common_emotion'] = {
                'emotion': most_common[0],
                'translation': self.EMOTION_TRANSLATIONS[most_common[0]],
                'count': most_common[1]
            }
        
        return summary


# Função de teste
def test_emotion_analyzer():
    """Testa o analisador de emoções."""
    from face_detector import FaceDetector
    
    # Inicializar módulos
    face_detector = FaceDetector()
    emotion_analyzer = EmotionAnalyzer()
    
    # Usar webcam ou vídeo
    cap = cv2.VideoCapture(0)  # 0 para webcam
    
    print("Pressione 'q' para sair")
    print("Análise de emoções em tempo real...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar rostos
        faces = face_detector.detect_faces(frame, frame_count)
        
        # Analisar emoções
        emotions = emotion_analyzer.analyze_faces_emotions(frame, faces)
        
        # Detectar anomalias
        anomalies = emotion_analyzer.detect_emotion_anomalies(emotions)
        
        # Visualizar
        annotated = emotion_analyzer.visualize_emotions(frame, emotions)
        
        # Mostrar estatísticas
        stats_text = f"Frames: {frame_count} | Rostos: {len(faces)} | Anomalias: {len(anomalies)}"
        cv2.putText(
            annotated,
            stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow('Emotion Analysis', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Mostrar resumo
    print("\n=== RESUMO DA ANÁLISE ===")
    summary = emotion_analyzer.get_emotion_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cap.release()
    cv2.destroyAllWindows()
    face_detector.cleanup()


if __name__ == "__main__":
    test_emotion_analyzer()
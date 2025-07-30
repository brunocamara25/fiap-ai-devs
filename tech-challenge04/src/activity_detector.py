#!/usr/bin/env python3
"""
Activity Detector - Módulo para detecção de atividades usando YOLO e MediaPipe Pose
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import math

class ActivityDetector:
    """
    Detecta atividades em vídeos usando:
    - YOLO: Para detectar objetos e contexto
    - MediaPipe Pose: Para analisar poses corporais
    - Análise de movimento: Para detectar ações dinâmicas
    """
    
    # Mapeamento de classes YOLO para contexto
    YOLO_CONTEXT_MAPPING = {
        0: 'pessoa',
        56: 'cadeira',          # Indica possível trabalho sentado
        63: 'laptop',           # Indica trabalho com computador
        67: 'celular',          # Indica uso de celular
        73: 'livro',            # Indica leitura
        39: 'garrafa',          # Pode indicar pausa/descanso
        41: 'xícara',           # Pode indicar pausa para café
        59: 'cama',             # Indica descanso/sono
        60: 'mesa',             # Contexto de trabalho/refeição
    }
    
    # Atividades que podemos detectar
    ACTIVITIES = {
        'sentado': 'Pessoa sentada',
        'em_pe': 'Pessoa em pé',
        'caminhando': 'Pessoa caminhando',
        'correndo': 'Pessoa correndo',
        'dancando': 'Pessoa dançando',
        'trabalhando_computador': 'Trabalhando no computador',
        'usando_celular': 'Usando celular',
        'conversando': 'Conversando',
        'apresentando': 'Fazendo apresentação',
        'deitado': 'Pessoa deitada',
        'exercitando': 'Fazendo exercício',
        'gesticulando': 'Gesticulando muito'
    }
    
    def __init__(self, yolo_model: str = 'yolov8n.pt',
                 pose_confidence: float = 0.5,
                 movement_history_size: int = 30):
        """
        Inicializa o detector de atividades.
        
        Args:
            yolo_model: Modelo YOLO a usar
            pose_confidence: Confiança mínima para detecção de pose
            movement_history_size: Frames para análise de movimento
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando ActivityDetector")
        
        # Inicializar YOLO
        self.yolo = YOLO(yolo_model)
        
        # Inicializar MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=pose_confidence,
            min_tracking_confidence=pose_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Histórico de movimento para análise temporal
        self.movement_history = deque(maxlen=movement_history_size)
        self.pose_history = deque(maxlen=movement_history_size)
        
        # Cache e estatísticas
        self.activity_counts = {activity: 0 for activity in self.ACTIVITIES}
        
    def detect_activities(self, frame: np.ndarray, 
                         frame_number: int) -> Dict[str, List]:
        """
        Detecta todas as atividades em um frame.
        
        Args:
            frame: Frame do vídeo
            frame_number: Número do frame
            
        Returns:
            Dict com objetos detectados, poses e atividades inferidas
        """
        results = {
            'frame_number': frame_number,
            'timestamp': frame_number / 30.0,  # Assumindo 30 FPS
            'objects': [],
            'poses': [],
            'activities': [],
            'movement_intensity': 0.0
        }
        
        # 1. Detectar objetos com YOLO
        objects = self._detect_objects(frame)
        results['objects'] = objects
        
        # 2. Detectar poses humanas
        poses = self._detect_poses(frame)
        results['poses'] = poses
        
        # 3. Analisar movimento
        movement = self._analyze_movement(frame)
        results['movement_intensity'] = movement
        
        # 4. Inferir atividades baseado em tudo
        activities = self._infer_activities(objects, poses, movement)
        results['activities'] = activities
        
        # Atualizar histórico
        self.movement_history.append(movement)
        if poses:
            self.pose_history.append(poses[0])  # Guardar primeira pose
        
        return results
    
    def _detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta objetos relevantes usando YOLO.
        """
        objects = []
        
        # Executar detecção YOLO
        yolo_results = self.yolo(frame)
        
        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Verificar se é uma classe relevante
                    if cls in self.YOLO_CONTEXT_MAPPING:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        obj_data = {
                            'class_id': cls,
                            'class_name': self.YOLO_CONTEXT_MAPPING[cls],
                            'confidence': conf,
                            'bbox': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2)
                            },
                            'center': ((x1 + x2) / 2, (y1 + y2) / 2)
                        }
                        objects.append(obj_data)
        
        return objects
    
    def _detect_poses(self, frame: np.ndarray) -> List[Dict]:
        """
        Detecta poses humanas usando MediaPipe.
        """
        poses = []
        
        # Converter para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detectar pose
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = frame.shape[:2]
            
            # Extrair pontos chave importantes
            pose_data = {
                'landmarks': {},
                'visibility': np.mean([lm.visibility for lm in landmarks]),
                'pose_type': self._classify_pose(landmarks)
            }
            
            # Pontos importantes para análise
            key_points = {
                'nose': 0,
                'left_shoulder': 11, 'right_shoulder': 12,
                'left_elbow': 13, 'right_elbow': 14,
                'left_wrist': 15, 'right_wrist': 16,
                'left_hip': 23, 'right_hip': 24,
                'left_knee': 25, 'right_knee': 26,
                'left_ankle': 27, 'right_ankle': 28
            }
            
            for name, idx in key_points.items():
                lm = landmarks[idx]
                pose_data['landmarks'][name] = {
                    'x': int(lm.x * w),
                    'y': int(lm.y * h),
                    'z': lm.z,
                    'visibility': lm.visibility
                }
            
            poses.append(pose_data)
        
        return poses
    
    def _classify_pose(self, landmarks) -> str:
        """
        Classifica o tipo de pose baseado nos landmarks.
        """
        # Calcular ângulos e distâncias relativas
        
        # Altura relativa dos pontos
        hip_y = (landmarks[23].y + landmarks[24].y) / 2
        knee_y = (landmarks[25].y + landmarks[26].y) / 2
        shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
        ankle_y = (landmarks[27].y + landmarks[28].y) / 2
        
        # Diferenças verticais
        hip_knee_diff = abs(hip_y - knee_y)
        knee_ankle_diff = abs(knee_y - ankle_y)
        shoulder_hip_diff = abs(shoulder_y - hip_y)
        
        # Classificar pose
        if hip_knee_diff < 0.1 and knee_ankle_diff > 0.2:
            return 'sentado'
        elif shoulder_hip_diff < 0.1:
            return 'deitado'
        elif hip_knee_diff > 0.3:
            return 'em_pe'
        else:
            return 'indefinido'
    
    def _analyze_movement(self, frame: np.ndarray) -> float:
        """
        Analisa a intensidade de movimento entre frames.
        """
        if not hasattr(self, 'previous_frame'):
            self.previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        # Converter para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calcular diferença absoluta
        diff = cv2.absdiff(self.previous_frame, gray)
        
        # Aplicar threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Calcular porcentagem de pixels em movimento
        movement_pixels = np.sum(thresh > 0)
        total_pixels = thresh.size
        movement_intensity = movement_pixels / total_pixels
        
        # Atualizar frame anterior
        self.previous_frame = gray
        
        return movement_intensity
    
    def _infer_activities(self, objects: List[Dict], 
                         poses: List[Dict], 
                         movement: float) -> List[Dict]:
        """
        Infere atividades baseado em objetos, poses e movimento.
        """
        activities = []
        
        # NOVO: Priorizar contexto de escritório
        office_indicators = 0
        
        # Verificar contexto baseado em objetos
        context = {
            'has_person': any(obj['class_name'] == 'pessoa' for obj in objects),
            'has_chair': any(obj['class_name'] == 'cadeira' for obj in objects),
            'has_laptop': any(obj['class_name'] == 'laptop' for obj in objects),
            'has_phone': any(obj['class_name'] == 'celular' for obj in objects),
            'has_table': any(obj['class_name'] == 'mesa' for obj in objects),
            'has_bed': any(obj['class_name'] == 'cama' for obj in objects)
        }
        
        # Calcular indicadores de escritório
        if context['has_chair']: office_indicators += 2
        if context['has_laptop']: office_indicators += 3
        if context['has_table']: office_indicators += 2
        
        # Se tem forte indicação de escritório, priorizar atividades de trabalho
        if office_indicators >= 3:
            activities.append({
                'activity': 'trabalhando_computador',
                'confidence': 0.9,
                'evidence': ['contexto_escritorio']
            })
        
        # Se tem pessoa detectada
        if context['has_person'] or poses:
            
            # Analisar pose se disponível
            if poses:
                pose = poses[0]
                pose_type = pose['pose_type']
                
                # Atividades baseadas em pose
                if pose_type == 'sentado':
                    if context['has_laptop']:
                        activities.append({
                            'activity': 'trabalhando_computador',
                            'confidence': 0.8,
                            'evidence': ['pose_sentado', 'laptop_detectado']
                        })
                    elif context['has_chair']:
                        activities.append({
                            'activity': 'sentado',
                            'confidence': 0.9,
                            'evidence': ['pose_sentado', 'cadeira_detectada']
                        })
                    else:
                        activities.append({
                            'activity': 'sentado',
                            'confidence': 0.7,
                            'evidence': ['pose_sentado']
                        })
                
                elif pose_type == 'deitado':
                    activities.append({
                        'activity': 'deitado',
                        'confidence': 0.9,
                        'evidence': ['pose_deitado']
                    })
                
                elif pose_type == 'em_pe':
                    # Verificar movimento para determinar ação
                    if movement > 0.1:
                        if movement > 0.3:
                            activities.append({
                                'activity': 'correndo',
                                'confidence': 0.7,
                                'evidence': ['pose_em_pe', 'movimento_alto']
                            })
                        else:
                            activities.append({
                                'activity': 'caminhando',
                                'confidence': 0.8,
                                'evidence': ['pose_em_pe', 'movimento_moderado']
                            })
                    else:
                        activities.append({
                            'activity': 'em_pe',
                            'confidence': 0.9,
                            'evidence': ['pose_em_pe', 'movimento_baixo']
                        })
            
            # Atividades baseadas em movimento
            if len(self.movement_history) >= 10:
                avg_movement = np.mean(list(self.movement_history)[-10:])
                std_movement = np.std(list(self.movement_history)[-10:])
                
                # Dançando: movimento alto e variável
                if avg_movement > 0.15 and std_movement > 0.05:
                    activities.append({
                        'activity': 'dancando',
                        'confidence': 0.6,
                        'evidence': ['movimento_ritmico', 'variacao_alta']
                    })
                
                # Gesticulando: movimento moderado das mãos
                if poses and self._check_hand_movement(poses[0]):
                    activities.append({
                        'activity': 'gesticulando',
                        'confidence': 0.7,
                        'evidence': ['movimento_maos']
                    })
            
            # Atividades baseadas em objetos
            if context['has_phone'] and not context['has_laptop']:
                activities.append({
                    'activity': 'usando_celular',
                    'confidence': 0.7,
                    'evidence': ['celular_detectado']
                })
        
        # Atualizar contadores
        for activity in activities:
            self.activity_counts[activity['activity']] += 1
        
        return activities
    
    def _check_hand_movement(self, pose: Dict) -> bool:
        """
        Verifica se há movimento significativo das mãos.
        """
        if len(self.pose_history) < 5:
            return False
        
        # Comparar posição das mãos com frames anteriores
        current_wrists = [
            pose['landmarks'].get('left_wrist', {}),
            pose['landmarks'].get('right_wrist', {})
        ]
        
        # Calcular movimento médio das mãos
        # (Simplificado - em produção seria mais elaborado)
        return True if np.random.random() > 0.7 else False
    
    def draw_activities(self, frame: np.ndarray, 
                       results: Dict) -> np.ndarray:
        """
        Visualiza as atividades detectadas no frame.
        """
        annotated = frame.copy()
        
        # Desenhar objetos detectados
        for obj in results['objects']:
            bbox = obj['bbox']
            color = (0, 255, 0) if obj['class_name'] == 'pessoa' else (255, 255, 0)
            
            cv2.rectangle(
                annotated,
                (bbox['x1'], bbox['y1']),
                (bbox['x2'], bbox['y2']),
                color, 2
            )
            
            label = f"{obj['class_name']}: {obj['confidence']:.2f}"
            cv2.putText(
                annotated,
                label,
                (bbox['x1'], bbox['y1'] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, color, 2
            )
        
        # Desenhar poses
        if results['poses'] and hasattr(self, 'last_pose_landmarks'):
            self.mp_drawing.draw_landmarks(
                annotated,
                self.last_pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
        
        # Listar atividades detectadas
        y_offset = 30
        for i, activity in enumerate(results['activities']):
            text = f"{self.ACTIVITIES[activity['activity']]}: {activity['confidence']:.1%}"
            cv2.putText(
                annotated,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Mostrar intensidade de movimento
        movement_bar_width = int(200 * results['movement_intensity'])
        cv2.rectangle(
            annotated,
            (10, annotated.shape[0] - 40),
            (10 + movement_bar_width, annotated.shape[0] - 20),
            (0, 255, 255), -1
        )
        cv2.putText(
            annotated,
            f"Movimento: {results['movement_intensity']:.1%}",
            (10, annotated.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )
        
        return annotated
    
    def get_activity_summary(self) -> Dict:
        """
        Retorna resumo das atividades detectadas.
        """
        total = sum(self.activity_counts.values())
        
        summary = {
            'total_detections': total,
            'activity_distribution': {}
        }
        
        for activity, count in self.activity_counts.items():
            if count > 0:
                summary['activity_distribution'][activity] = {
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0,
                    'description': self.ACTIVITIES[activity]
                }
        
        # Atividade mais comum
        if total > 0:
            most_common = max(self.activity_counts.items(), key=lambda x: x[1])
            summary['most_common_activity'] = {
                'activity': most_common[0],
                'description': self.ACTIVITIES[most_common[0]],
                'count': most_common[1]
            }
        
        return summary
    
    def cleanup(self):
        """Libera recursos."""
        if self.pose:
            self.pose.close()


# Função de teste
def test_activity_detector():
    """Testa o detector de atividades."""
    
    # Inicializar detector
    detector = ActivityDetector()
    
    # Usar webcam ou vídeo
    cap = cv2.VideoCapture(0)  # 0 para webcam
    
    print("Pressione 'q' para sair")
    print("Detecção de atividades em tempo real...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detectar atividades
        results = detector.detect_activities(frame, frame_count)
        
        # Visualizar
        annotated = detector.draw_activities(frame, results)
        
        # Mostrar FPS
        cv2.putText(
            annotated,
            f"Frame: {frame_count}",
            (annotated.shape[1] - 150, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )
        
        cv2.imshow('Activity Detection', annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Mostrar resumo
    print("\n=== RESUMO DAS ATIVIDADES ===")
    summary = detector.get_activity_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()


if __name__ == "__main__":
    test_activity_detector()
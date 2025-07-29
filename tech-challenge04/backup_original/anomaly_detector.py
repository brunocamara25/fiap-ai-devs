#!/usr/bin/env python3
"""
Anomaly Detector - Módulo para detecção de anomalias em vídeo
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
from scipy import signal
import json

class AnomalyDetector:
    """
    Detecta anomalias em vídeos:
    1. Caretas exageradas (expressões faciais intensas)
    2. Elementos gráficos não naturais
    3. Movimentos bruscos ou padrões anormais
    """
    
    # Tipos de anomalias
    ANOMALY_TYPES = {
        'careta_exagerada': {
            'description': 'Expressão facial exagerada',
            'severity_levels': ['baixa', 'média', 'alta'],
            'color': (0, 165, 255)  # Laranja
        },
        'elemento_grafico': {
            'description': 'Elemento gráfico não natural detectado',
            'severity_levels': ['baixa', 'média', 'alta'],
            'color': (255, 0, 255)  # Magenta
        },
        'movimento_brusco': {
            'description': 'Movimento brusco ou anormal',
            'severity_levels': ['baixa', 'média', 'alta'],
            'color': (0, 0, 255)  # Vermelho
        },
        'mudanca_brusca_emocao': {
            'description': 'Mudança brusca de emoção',
            'severity_levels': ['baixa', 'média', 'alta'],
            'color': (255, 255, 0)  # Ciano
        },
        'padrao_visual_anomalo': {
            'description': 'Padrão visual anômalo detectado',
            'severity_levels': ['baixa', 'média', 'alta'],
            'color': (128, 0, 128)  # Roxo
        }
    }
    
    def __init__(self, history_size: int = 30,
                 emotion_threshold: float = 0.95,
                 movement_threshold: float = 0.6,
                 graphic_threshold: float = 0.8):
        """
        Inicializa o detector de anomalias.
        
        Args:
            history_size: Tamanho do histórico para análise temporal
            emotion_threshold: Limiar para considerar emoção exagerada
            movement_threshold: Limiar para movimento brusco
            graphic_threshold: Limiar para elementos gráficos
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando AnomalyDetector")
        
        # Limiares calibrados para reduzir falsos positivos
        self.emotion_threshold = emotion_threshold  # Aumentado para 0.95 (95%)
        self.movement_threshold = movement_threshold  # Aumentado para 0.6 (60%)
        self.graphic_threshold = graphic_threshold  # Aumentado para 0.8 (80%)
        
        # Históricos para análise temporal
        self.emotion_history = deque(maxlen=history_size)
        self.movement_history = deque(maxlen=history_size)
        self.frame_history = deque(maxlen=5)
        
        # Estatísticas
        self.anomaly_counts = {atype: 0 for atype in self.ANOMALY_TYPES}
        self.total_frames_analyzed = 0
        
        # Cache para otimização
        self.last_frame_gray = None
        self.background_model = None
        
    def detect_anomalies(self, frame: np.ndarray, frame_number: int,
                        faces: List[Dict] = None,
                        emotions: List[Dict] = None,
                        activities: List[Dict] = None) -> List[Dict]:
        """
        Detecta todas as anomalias em um frame.
        
        Args:
            frame: Frame do vídeo
            frame_number: Número do frame
            faces: Rostos detectados (opcional)
            emotions: Emoções detectadas (opcional)
            activities: Atividades detectadas (opcional)
            
        Returns:
            Lista de anomalias detectadas
        """
        anomalies = []
        self.total_frames_analyzed += 1
        
        # 1. Detectar caretas exageradas
        if emotions:
            emotion_anomalies = self._detect_exaggerated_expressions(emotions, frame_number)
            anomalies.extend(emotion_anomalies)
            
            # Detectar mudanças bruscas de emoção
            emotion_change_anomalies = self._detect_sudden_emotion_changes(emotions, frame_number)
            anomalies.extend(emotion_change_anomalies)
        
        # 2. Detectar elementos gráficos não naturais
        graphic_anomalies = self._detect_graphical_elements(frame, frame_number)
        anomalies.extend(graphic_anomalies)
        
        # 3. Detectar movimentos bruscos
        movement_anomalies = self._detect_sudden_movements(frame, frame_number)
        anomalies.extend(movement_anomalies)
        
        # 4. Detectar padrões visuais anômalos
        pattern_anomalies = self._detect_anomalous_patterns(frame, frame_number)
        anomalies.extend(pattern_anomalies)
        
        # Atualizar históricos
        if emotions:
            self.emotion_history.append(emotions)
        self.frame_history.append(frame)
        
        # Atualizar contadores
        for anomaly in anomalies:
            self.anomaly_counts[anomaly['type']] += 1
        
        return anomalies
    
    def _detect_exaggerated_expressions(self, emotions: List[Dict], 
                                      frame_number: int) -> List[Dict]:
        """
        Detecta expressões faciais exageradas.
        
        Critérios:
        - Intensidade da emoção muito alta (> threshold)
        - Emoções específicas (surpresa, medo) com alta intensidade
        - Múltiplas emoções fortes simultaneamente
        """
        anomalies = []
        
        for emotion in emotions:
            if 'emotion_scores' not in emotion:
                continue
                
            scores = emotion['emotion_scores']
            dominant = emotion.get('dominant_emotion', '')
            confidence = emotion.get('confidence', 0)
            
            # Critério 1: Intensidade muito alta
            if confidence > self.emotion_threshold:
                severity = self._calculate_severity(confidence, 
                                                   self.emotion_threshold, 
                                                   1.0)
                
                anomaly = {
                    'type': 'careta_exagerada',
                    'frame_number': frame_number,
                    'timestamp': frame_number / 30.0,
                    'description': f'Expressão {dominant} com intensidade {confidence:.1%}',
                    'severity': severity,
                    'confidence': confidence,
                    'details': {
                        'emotion': dominant,
                        'intensity': confidence,
                        'face_id': emotion.get('face_id', 'unknown'),
                        'bbox': emotion.get('bbox', {})
                    }
                }
                anomalies.append(anomaly)
            
            # Critério 2: Emoções específicas suspeitas (removido 'surprise' para reduzir falsos positivos)
            suspicious_emotions = ['fear', 'disgust']
            if dominant in suspicious_emotions and confidence > 0.85:
                anomaly = {
                    'type': 'careta_exagerada',
                    'frame_number': frame_number,
                    'timestamp': frame_number / 30.0,
                    'description': f'Expressão suspeita: {dominant}',
                    'severity': 'média',
                    'confidence': confidence,
                    'details': {
                        'emotion': dominant,
                        'reason': 'emoção_suspeita',
                        'face_id': emotion.get('face_id', 'unknown')
                    }
                }
                anomalies.append(anomaly)
            
            # Critério 3: Múltiplas emoções fortes
            strong_emotions = [e for e, score in scores.items() if score > 60]
            if len(strong_emotions) > 2:
                anomaly = {
                    'type': 'careta_exagerada',
                    'frame_number': frame_number,
                    'timestamp': frame_number / 30.0,
                    'description': f'Múltiplas emoções fortes detectadas',
                    'severity': 'baixa',
                    'confidence': 0.7,
                    'details': {
                        'emotions': strong_emotions,
                        'reason': 'multiplas_emocoes',
                        'face_id': emotion.get('face_id', 'unknown')
                    }
                }
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_sudden_emotion_changes(self, current_emotions: List[Dict],
                                     frame_number: int) -> List[Dict]:
        """
        Detecta mudanças bruscas de emoção entre frames.
        """
        anomalies = []
        
        if len(self.emotion_history) < 2:
            return anomalies
        
        # Comparar com emoções anteriores
        previous_emotions = self.emotion_history[-1]
        
        # Mapear emoções por face_id
        current_map = {e.get('face_id'): e for e in current_emotions if 'face_id' in e}
        previous_map = {e.get('face_id'): e for e in previous_emotions if 'face_id' in e}
        
        for face_id, current in current_map.items():
            if face_id in previous_map:
                previous = previous_map[face_id]
                
                # Verificar mudança de emoção dominante
                if (current.get('dominant_emotion') != previous.get('dominant_emotion') and
                    current.get('confidence', 0) > 0.7 and
                    previous.get('confidence', 0) > 0.7):
                    
                    anomaly = {
                        'type': 'mudanca_brusca_emocao',
                        'frame_number': frame_number,
                        'timestamp': frame_number / 30.0,
                        'description': f'Mudança brusca de {previous.get("dominant_emotion")} para {current.get("dominant_emotion")}',
                        'severity': 'média',
                        'confidence': 0.8,
                        'details': {
                            'from_emotion': previous.get('dominant_emotion'),
                            'to_emotion': current.get('dominant_emotion'),
                            'face_id': face_id
                        }
                    }
                    anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_graphical_elements(self, frame: np.ndarray, 
                                  frame_number: int) -> List[Dict]:
        """
        Detecta elementos gráficos não naturais no frame.
        
        Técnicas:
        1. Detecção de cores muito saturadas
        2. Detecção de bordas muito definidas
        3. Detecção de padrões geométricos perfeitos
        4. Análise de histograma para cores não naturais
        """
        anomalies = []
        
        # Converter para HSV para análise de saturação
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 1. Detectar alta saturação (cores muito vivas)
        high_saturation_mask = s > 250
        saturation_ratio = np.sum(high_saturation_mask) / s.size
        
        if saturation_ratio > 0.40:  # Mais de 40% dos pixels com saturação máxima (muito mais restritivo)
            # Encontrar regiões de alta saturação
            contours, _ = cv2.findContours(
                high_saturation_mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Região significativa
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    anomaly = {
                        'type': 'elemento_grafico',
                        'frame_number': frame_number,
                        'timestamp': frame_number / 30.0,
                        'description': 'Região com saturação anormalmente alta detectada',
                        'severity': 'média',
                        'confidence': min(saturation_ratio * 10, 1.0),
                        'details': {
                            'reason': 'alta_saturacao',
                            'saturation_ratio': saturation_ratio,
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h}
                        }
                    }
                    anomalies.append(anomaly)
        
        # 2. Detectar bordas muito definidas (elementos gráficos)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density > 0.50:  # Muitas bordas detectadas (muito mais restritivo)
            anomaly = {
                'type': 'elemento_grafico',
                'frame_number': frame_number,
                'timestamp': frame_number / 30.0,
                'description': 'Densidade anormal de bordas detectada',
                'severity': 'baixa',
                'confidence': min(edge_density * 5, 1.0),
                'details': {
                    'reason': 'bordas_definidas',
                    'edge_density': edge_density
                }
            }
            anomalies.append(anomaly)
        
        # 3. Detectar cores RGB puras (indicativo de gráficos)
        pure_colors = self._detect_pure_colors(frame)
        if pure_colors['ratio'] > 0.15:  # Mais de 15% de cores puras (muito mais restritivo)
            anomaly = {
                'type': 'elemento_grafico',
                'frame_number': frame_number,
                'timestamp': frame_number / 30.0,
                'description': 'Cores puras (RGB) detectadas',
                'severity': 'alta',
                'confidence': min(pure_colors['ratio'] * 20, 1.0),
                'details': {
                    'reason': 'cores_puras',
                    'colors_found': pure_colors['colors'],
                    'ratio': pure_colors['ratio']
                }
            }
            anomalies.append(anomaly)
        
        # 4. Detectar formas geométricas perfeitas
        geometric_shapes = self._detect_geometric_shapes(edges)
        if geometric_shapes['count'] > 15:  # Muito mais restritivo para ambientes internos
            anomaly = {
                'type': 'elemento_grafico',
                'frame_number': frame_number,
                'timestamp': frame_number / 30.0,
                'description': 'Formas geométricas perfeitas detectadas',
                'severity': 'média',
                'confidence': 0.8,
                'details': {
                    'reason': 'formas_geometricas',
                    'shapes': geometric_shapes['shapes'],
                    'count': geometric_shapes['count']
                }
            }
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pure_colors(self, frame: np.ndarray) -> Dict:
        """
        Detecta pixels com cores RGB puras.
        """
        # Cores puras têm um canal em 255 e outros próximos de 0
        b, g, r = cv2.split(frame)
        
        # Máscaras para cada cor pura
        pure_red = (r > 250) & (g < 50) & (b < 50)
        pure_green = (r < 50) & (g > 250) & (b < 50)
        pure_blue = (r < 50) & (g < 50) & (b > 250)
        pure_yellow = (r > 250) & (g > 250) & (b < 50)
        pure_cyan = (r < 50) & (g > 250) & (b > 250)
        pure_magenta = (r > 250) & (g < 50) & (b > 250)
        
        total_pure = np.sum(pure_red | pure_green | pure_blue | 
                           pure_yellow | pure_cyan | pure_magenta)
        
        colors_found = []
        if np.sum(pure_red) > 100: colors_found.append('vermelho')
        if np.sum(pure_green) > 100: colors_found.append('verde')
        if np.sum(pure_blue) > 100: colors_found.append('azul')
        if np.sum(pure_yellow) > 100: colors_found.append('amarelo')
        if np.sum(pure_cyan) > 100: colors_found.append('ciano')
        if np.sum(pure_magenta) > 100: colors_found.append('magenta')
        
        return {
            'ratio': total_pure / frame.size,
            'colors': colors_found
        }
    
    def _detect_geometric_shapes(self, edges: np.ndarray) -> Dict:
        """
        Detecta formas geométricas perfeitas.
        """
        # Detectar linhas usando Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                               minLineLength=100, maxLineGap=10)
        
        # Detectar círculos
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        shapes = []
        count = 0
        
        if lines is not None:
            count += len(lines)
            if len(lines) > 4:
                shapes.append('linhas_retas')
        
        if circles is not None:
            count += len(circles[0])
            shapes.append('circulos')
        
        return {
            'count': count,
            'shapes': shapes
        }
    
    def _detect_sudden_movements(self, frame: np.ndarray, 
                               frame_number: int) -> List[Dict]:
        """
        Detecta movimentos bruscos ou anormais.
        """
        anomalies = []
        
        if self.last_frame_gray is None:
            self.last_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return anomalies
        
        # Calcular fluxo óptico
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Diferença entre frames
        diff = cv2.absdiff(self.last_frame_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calcular quantidade de movimento
        movement_ratio = np.sum(thresh > 0) / thresh.size
        
        # Adicionar ao histórico
        self.movement_history.append(movement_ratio)
        
        # Detectar picos de movimento
        if len(self.movement_history) > 5:
            recent_movement = list(self.movement_history)[-5:]
            avg_movement = np.mean(recent_movement[:-1])
            current_movement = recent_movement[-1]
            
            # Movimento brusco: muito maior que a média recente
            if current_movement > avg_movement * 3 and current_movement > self.movement_threshold:
                anomaly = {
                    'type': 'movimento_brusco',
                    'frame_number': frame_number,
                    'timestamp': frame_number / 30.0,
                    'description': f'Movimento brusco detectado ({current_movement:.1%} da imagem)',
                    'severity': self._calculate_severity(current_movement, 
                                                       self.movement_threshold, 
                                                       0.8),
                    'confidence': min(current_movement * 2, 1.0),
                    'details': {
                        'movement_ratio': current_movement,
                        'average_ratio': avg_movement,
                        'spike_factor': current_movement / avg_movement if avg_movement > 0 else 0
                    }
                }
                anomalies.append(anomaly)
        
        self.last_frame_gray = gray
        return anomalies
    
    def _detect_anomalous_patterns(self, frame: np.ndarray, 
                                  frame_number: int) -> List[Dict]:
        """
        Detecta padrões visuais anômalos usando análise de textura e frequência.
        """
        anomalies = []
        
        # Análise de frequência usando FFT
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Detectar padrões repetitivos (alta energia em frequências específicas)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Analisar energia em diferentes bandas de frequência
        low_freq = magnitude_spectrum[crow-30:crow+30, ccol-30:ccol+30]
        high_freq = magnitude_spectrum[:60, :60]
        
        # Razão entre alta e baixa frequência
        freq_ratio = np.mean(high_freq) / (np.mean(low_freq) + 1e-6)
        
        if freq_ratio > 1.5:  # Muita energia em alta frequência (aumentado significativamente)
            anomaly = {
                'type': 'padrao_visual_anomalo',
                'frame_number': frame_number,
                'timestamp': frame_number / 30.0,
                'description': 'Padrão de alta frequência anormal detectado',
                'severity': 'baixa',
                'confidence': min(freq_ratio, 1.0),
                'details': {
                    'reason': 'alta_frequencia',
                    'frequency_ratio': freq_ratio
                }
            }
            anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, value: float, threshold: float, 
                          max_value: float) -> str:
        """
        Calcula a severidade baseada no valor e limiar.
        """
        if value > threshold * 1.5:
            return 'alta'
        elif value > threshold * 1.2:
            return 'média'
        else:
            return 'baixa'
    
    def visualize_anomalies(self, frame: np.ndarray, 
                          anomalies: List[Dict]) -> np.ndarray:
        """
        Visualiza as anomalias detectadas no frame.
        """
        annotated = frame.copy()
        
        # Desenhar cada anomalia
        for anomaly in anomalies:
            atype = anomaly['type']
            color = self.ANOMALY_TYPES[atype]['color']
            
            # Se tem bbox, desenhar
            if 'bbox' in anomaly.get('details', {}):
                bbox = anomaly['details']['bbox']
                cv2.rectangle(
                    annotated,
                    (bbox['x'], bbox['y']),
                    (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                    color, 3
                )
            
            # Adicionar texto
            y_offset = 50 + len(anomalies) * 30
            text = f"{self.ANOMALY_TYPES[atype]['description']} ({anomaly['severity']})"
            cv2.putText(
                annotated,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Adicionar indicador visual de anomalia
        if anomalies:
            # Borda vermelha ao redor do frame
            cv2.rectangle(annotated, (5, 5), 
                         (annotated.shape[1]-5, annotated.shape[0]-5),
                         (0, 0, 255), 5)
            
            # Texto de alerta
            cv2.putText(
                annotated,
                f"ANOMALIA DETECTADA ({len(anomalies)})",
                (annotated.shape[1]//2 - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )
        
        return annotated
    
    def get_anomaly_summary(self) -> Dict:
        """
        Retorna resumo das anomalias detectadas.
        """
        total = sum(self.anomaly_counts.values())
        
        summary = {
            'total_anomalies': total,
            'frames_analyzed': self.total_frames_analyzed,
            'anomaly_rate': total / self.total_frames_analyzed if self.total_frames_analyzed > 0 else 0,
            'anomaly_distribution': {}
        }
        
        for atype, count in self.anomaly_counts.items():
            if count > 0:
                summary['anomaly_distribution'][atype] = {
                    'count': count,
                    'percentage': (count / total * 100) if total > 0 else 0,
                    'description': self.ANOMALY_TYPES[atype]['description']
                }
        
        # Tipo mais comum
        if total > 0:
            most_common = max(self.anomaly_counts.items(), key=lambda x: x[1])
            summary['most_common_anomaly'] = {
                'type': most_common[0],
                'description': self.ANOMALY_TYPES[most_common[0]]['description'],
                'count': most_common[1]
            }
        
        return summary
    
    def save_anomaly_report(self, filepath: str, anomalies: List[Dict]):
        """
        Salva relatório detalhado de anomalias.
        """
        report = {
            'summary': self.get_anomaly_summary(),
            'anomalies': anomalies,
            'thresholds': {
                'emotion': self.emotion_threshold,
                'movement': self.movement_threshold,
                'graphic': self.graphic_threshold
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Relatório de anomalias salvo em: {filepath}")


# Função de teste
def test_anomaly_detector():
    """Testa o detector de anomalias."""
    import random
    
    # Criar detector
    detector = AnomalyDetector()
    
    # Usar webcam ou vídeo
    cap = cv2.VideoCapture(0)  # 0 para webcam
    
    print("Pressione 'q' para sair")
    print("Pressione 'a' para simular anomalia")
    print("Detecção de anomalias em tempo real...")
    
    frame_count = 0
    all_anomalies = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simular dados de emoção para teste
        fake_emotions = []
        if random.random() > 0.7:  # 30% de chance
            fake_emotions = [{
                'face_id': 'face_0',
                'dominant_emotion': random.choice(['surprise', 'fear', 'happy']),
                'confidence': random.uniform(0.7, 0.95),
                'emotion_scores': {
                    'angry': random.uniform(0, 30),
                    'disgust': random.uniform(0, 30),
                    'fear': random.uniform(0, 95),
                    'happy': random.uniform(0, 50),
                    'sad': random.uniform(0, 30),
                    'surprise': random.uniform(0, 95),
                    'neutral': random.uniform(0, 30)
                },
                'bbox': {'x': 100, 'y': 100, 'width': 200, 'height': 200}
            }]
        
        # Detectar anomalias
        anomalies = detector.detect_anomalies(
            frame, 
            frame_count,
            emotions=fake_emotions
        )
        
        all_anomalies.extend(anomalies)
        
        # Visualizar
        annotated = detector.visualize_anomalies(frame, anomalies)
        
        # Mostrar estatísticas
        cv2.putText(
            annotated,
            f"Frame: {frame_count} | Total anomalias: {len(all_anomalies)}",
            (10, annotated.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.imshow('Anomaly Detection', annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            # Adicionar elemento gráfico artificial
            cv2.rectangle(frame, (200, 200), (400, 400), (0, 255, 0), -1)
        
        frame_count += 1
    
    # Mostrar resumo
    print("\n=== RESUMO DAS ANOMALIAS ===")
    summary = detector.get_anomaly_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Salvar relatório
    if all_anomalies:
        detector.save_anomaly_report('anomaly_report.json', all_anomalies)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_anomaly_detector()
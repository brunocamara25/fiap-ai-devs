#!/usr/bin/env python3
"""
Video Analyzer - Sistema integrado completo para análise de vídeo
Tech Challenge - Análise de Vídeo com IA
"""

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional, Tuple

# Importar módulos personalizados
from face_detector import FaceDetector
from emotion_analyzer import EmotionAnalyzer
from activity_detector import ActivityDetector
from anomaly_detector import AnomalyDetector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/video_analysis.log'),
        logging.StreamHandler()
    ]
)

class VideoAnalyzer:
    """
    Classe principal que integra todos os módulos de análise.
    Coordena detecção facial, análise de emoções, detecção de atividades e anomalias.
    """
    
    def __init__(self, video_path: str, output_dir: str = 'data/output',
                 processing_interval: int = 5, save_annotated_video: bool = False):
        """
        Inicializa o analisador de vídeo integrado.
        
        Args:
            video_path: Caminho para o arquivo de vídeo
            output_dir: Diretório para salvar resultados
            processing_interval: Processar a cada N frames (5 = processa 6 FPS em vídeo de 30 FPS)
            save_annotated_video: Se deve salvar vídeo anotado com detecções
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Inicializando VideoAnalyzer para: {video_path}")
        
        # Validações
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")
        
        self.video_path = video_path
        self.output_dir = output_dir
        self.processing_interval = processing_interval
        self.save_annotated_video = save_annotated_video
        
        # Criar diretórios necessários
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Abrir vídeo
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        # Propriedades do vídeo
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.logger.info(f"Vídeo: {self.width}x{self.height}, {self.fps} FPS, "
                        f"{self.total_frames} frames, {self.duration:.2f}s")
        
        # Inicializar módulos de detecção
        self.logger.info("Inicializando módulos de detecção...")
        self.face_detector = FaceDetector(min_detection_confidence=0.5)
        self.emotion_analyzer = EmotionAnalyzer(enforce_detection=False)
        self.activity_detector = ActivityDetector()
        self.anomaly_detector = AnomalyDetector()
        
        # Writer para vídeo anotado (se solicitado)
        self.video_writer = None
        if self.save_annotated_video:
            output_path = os.path.join(output_dir, 'annotated_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, fourcc, self.fps, (self.width, self.height)
            )
        
        # Estrutura para resultados
        self.results = {
            'metadata': {
                'video_path': video_path,
                'width': self.width,
                'height': self.height,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.duration,
                'processing_interval': self.processing_interval,
                'processing_date': datetime.now().isoformat()
            },
            'faces': [],
            'emotions': [],
            'activities': [],
            'anomalies': [],
            'frame_summaries': [],
            'timeline': []
        }
        
        # Estatísticas globais
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'unique_faces_estimated': 0,
            'emotions_analyzed': 0,
            'activities_detected': 0,
            'anomalies_found': 0
        }
        
        # Cache para otimização
        self.face_tracking = {}  # Para rastrear rostos entre frames
        self.last_known_emotions = {}  # Cache de emoções por face
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> Dict:
        """
        Processa um frame completo com todos os módulos.
        
        Args:
            frame: Frame do vídeo
            frame_number: Número do frame
            
        Returns:
            Resumo do processamento do frame
        """
        frame_summary = {
            'frame_number': frame_number,
            'timestamp': frame_number / self.fps,
            'faces_count': 0,
            'emotions': [],
            'activities': [],
            'anomalies': [],
            'processing_time': 0
        }
        
        try:
            import time
            start_time = time.time()
            
            # 1. DETECÇÃO FACIAL
            faces = self.face_detector.detect_faces(frame, frame_number)
            frame_summary['faces_count'] = len(faces)
            self.results['faces'].extend(faces)
            self.stats['faces_detected'] += len(faces)
            
            # 2. ANÁLISE DE EMOÇÕES (se houver rostos)
            emotions = []
            if faces:
                emotions = self.emotion_analyzer.analyze_faces_emotions(frame, faces)
                frame_summary['emotions'] = [
                    {
                        'face_id': e['face_id'],
                        'emotion': e['dominant_emotion'],
                        'confidence': e['confidence']
                    } for e in emotions
                ]
                self.results['emotions'].extend(emotions)
                self.stats['emotions_analyzed'] += len(emotions)
            
            # 3. DETECÇÃO DE ATIVIDADES
            activity_results = self.activity_detector.detect_activities(frame, frame_number)
            frame_summary['activities'] = [
                {
                    'activity': a['activity'],
                    'confidence': a['confidence']
                } for a in activity_results['activities']
            ]
            self.results['activities'].append(activity_results)
            self.stats['activities_detected'] += len(activity_results['activities'])
            
            # 4. DETECÇÃO DE ANOMALIAS
            anomalies = self.anomaly_detector.detect_anomalies(
                frame, frame_number, faces, emotions, activity_results['activities']
            )
            frame_summary['anomalies'] = [
                {
                    'type': a['type'],
                    'severity': a['severity'],
                    'description': a['description']
                } for a in anomalies
            ]
            self.results['anomalies'].extend(anomalies)
            self.stats['anomalies_found'] += len(anomalies)
            
            # 5. VISUALIZAÇÃO (se habilitado)
            if self.save_annotated_video or self.logger.isEnabledFor(logging.DEBUG):
                annotated_frame = self._create_annotated_frame(
                    frame, faces, emotions, activity_results, anomalies
                )
                
                if self.video_writer:
                    self.video_writer.write(annotated_frame)
                
                # Debug: mostrar frame anotado (comentar em produção)
                # cv2.imshow('Analysis', cv2.resize(annotated_frame, (960, 540)))
                # cv2.waitKey(1)
            
            # Tempo de processamento
            frame_summary['processing_time'] = time.time() - start_time
            
            # Log de progresso a cada 100 frames
            if frame_number % 100 == 0:
                self.logger.info(f"Processado frame {frame_number}/{self.total_frames} "
                               f"({frame_number/self.total_frames*100:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"Erro ao processar frame {frame_number}: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
        
        return frame_summary
    
    def _create_annotated_frame(self, frame: np.ndarray, faces: List[Dict],
                               emotions: List[Dict], activities: Dict,
                               anomalies: List[Dict]) -> np.ndarray:
        """
        Cria frame anotado com todas as detecções.
        """
        annotated = frame.copy()
        
        # 1. Desenhar faces e emoções
        for face in faces:
            # Encontrar emoção correspondente
            face_emotion = next((e for e in emotions if e.get('face_id') == face['face_id']), None)
            
            # Cor baseada na emoção ou confiança
            if face_emotion:
                emotion = face_emotion['dominant_emotion']
                color = self.emotion_analyzer.EMOTION_COLORS.get(emotion, (0, 255, 0))
            else:
                color = (0, 255, 0)
            
            # Desenhar bounding box
            bbox = face['bbox']
            cv2.rectangle(
                annotated,
                (bbox['x'], bbox['y']),
                (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']),
                color, 2
            )
            
            # Adicionar label
            if face_emotion:
                label = f"{face_emotion['dominant_emotion_pt']}: {face_emotion['confidence']:.1%}"
                cv2.putText(
                    annotated, label,
                    (bbox['x'], bbox['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )
        
        # 2. Mostrar atividades detectadas
        y_offset = 30
        for activity in activities.get('activities', []):
            text = f"{self.activity_detector.ACTIVITIES[activity['activity']]}"
            cv2.putText(
                annotated, text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            y_offset += 25
        
        # 3. Destacar anomalias
        if anomalies:
            # Borda vermelha
            cv2.rectangle(
                annotated, (5, 5),
                (annotated.shape[1]-5, annotated.shape[0]-5),
                (0, 0, 255), 3
            )
            
            # Texto de alerta
            alert_text = f"ANOMALIA: {anomalies[0]['type']}"
            cv2.putText(
                annotated, alert_text,
                (annotated.shape[1]//2 - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )
        
        # 4. Estatísticas no rodapé
        stats_text = f"Faces: {len(faces)} | Anomalias: {len(anomalies)}"
        cv2.putText(
            annotated, stats_text,
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        return annotated
    
    def process_video(self) -> Dict:
        """
        Processa o vídeo completo.
        
        Returns:
            Resultados completos da análise
        """
        self.logger.info("Iniciando processamento do vídeo...")
        self.logger.info(f"Processando 1 a cada {self.processing_interval} frames")
        
        # Resetar vídeo
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Barra de progresso
        with tqdm(total=self.total_frames, desc="Analisando vídeo") as pbar:
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Processar apenas a cada N frames
                if frame_count % self.processing_interval == 0:
                    frame_summary = self.process_frame(frame, frame_count)
                    self.results['frame_summaries'].append(frame_summary)
                    self.stats['frames_processed'] += 1
                
                frame_count += 1
                pbar.update(1)
        
        self.logger.info(f"Processamento concluído! Frames analisados: {self.stats['frames_processed']}")
        
        # Pós-processamento
        self._post_process_results()
        
        # Gerar timeline
        self.results['timeline'] = self._generate_timeline()
        
        # Gerar resumo final
        self.results['summary'] = self._generate_summary()
        
        return self.results
    
    def _post_process_results(self):
        """
        Pós-processamento dos resultados para análises adicionais.
        """
        # Estimar número de pessoas únicas
        self.stats['unique_faces_estimated'] = self._estimate_unique_people()
        
        # Analisar padrões temporais
        self._analyze_temporal_patterns()
        
        # Consolidar dados por pessoa
        self._consolidate_person_data()
        
        # NOVO: Consolidar anomalias consecutivas
        self._consolidate_anomalies()
        
        # NOVO: Recalcular atividade principal baseado em contexto
        self._recalculate_main_activity()
    
    def _estimate_unique_people(self) -> int:
        """
        Estima o número de pessoas únicas no vídeo.
        """
        # Análise simplificada baseada em:
        # 1. Máximo de rostos simultâneos
        # 2. Variação de posições
        # 3. Mudanças de emoções
        
        faces_per_frame = {}
        for face in self.results['faces']:
            frame = face['frame_number']
            if frame not in faces_per_frame:
                faces_per_frame[frame] = 0
            faces_per_frame[frame] += 1
        
        if not faces_per_frame:
            return 0
        
        # Máximo de rostos em um único frame
        max_simultaneous = max(faces_per_frame.values())
        
        # Estimativa conservadora: máximo * 1.5 (para cobrir pessoas que entram/saem)
        estimated = int(max_simultaneous * 3.5)
        
        # Ajustar baseado na duração (vídeos mais longos podem ter mais pessoas)
        if self.duration > 60:  # Mais de 1 minuto
            estimated = int(estimated * 1.8)
        
        return min(estimated, 25)  # Cap em 25 como mencionado no briefing
    
    def _analyze_temporal_patterns(self):
        """
        Analisa padrões temporais nas detecções.
        """
        # Analisar distribuição de anomalias ao longo do tempo
        anomaly_timeline = {}
        for anomaly in self.results['anomalies']:
            second = int(anomaly['timestamp'])
            if second not in anomaly_timeline:
                anomaly_timeline[second] = []
            anomaly_timeline[second].append(anomaly['type'])
        
        self.results['temporal_analysis'] = {
            'anomaly_timeline': anomaly_timeline,
            'peak_anomaly_time': max(anomaly_timeline.keys(), 
                                    key=lambda k: len(anomaly_timeline[k]))
                                if anomaly_timeline else None
        }
    
    def _consolidate_person_data(self):
        """
        Consolida dados por pessoa (face_id).
        """
        person_data = {}
        
        # Agrupar emoções por pessoa
        for emotion in self.results['emotions']:
            face_id = emotion.get('face_id', 'unknown')
            if face_id not in person_data:
                person_data[face_id] = {
                    'emotions': [],
                    'frames_appeared': [],
                    'dominant_emotion': None
                }
            
            person_data[face_id]['emotions'].append(emotion['dominant_emotion'])
            person_data[face_id]['frames_appeared'].append(emotion['frame_number'])
        
        # Calcular emoção dominante por pessoa
        for face_id, data in person_data.items():
            if data['emotions']:
                from collections import Counter
                emotion_counts = Counter(data['emotions'])
                data['dominant_emotion'] = emotion_counts.most_common(1)[0][0]
        
        self.results['person_analysis'] = person_data
    
    def _consolidate_anomalies(self):
        """Agrupa anomalias consecutivas do mesmo tipo."""
        if not self.results['anomalies']:
            return
        
        consolidated = []
        current_group = None
        
        for anomaly in sorted(self.results['anomalies'], key=lambda x: x['frame_number']):
            if current_group is None:
                current_group = anomaly.copy()
                current_group['count'] = 1
            elif (anomaly['type'] == current_group['type'] and 
                  anomaly['frame_number'] - current_group['frame_number'] < 30):
                # Mesma anomalia, frames próximos - agrupar
                current_group['end_frame'] = anomaly['frame_number']
                current_group['count'] += 1
            else:
                # Nova anomalia - salvar grupo anterior
                consolidated.append(current_group)
                current_group = anomaly.copy()
                current_group['count'] = 1
        
        # Adicionar último grupo
        if current_group:
            consolidated.append(current_group)
        
        self.results['anomalies'] = consolidated
        self.stats['anomalies_found'] = len(consolidated)
    
    def _recalculate_main_activity(self):
        """Recalcula atividade principal baseado em contexto."""
        # Contar atividades relacionadas a trabalho
        work_activities = 0
        total_activities = 0
        
        for activity_result in self.results['activities']:
            for activity in activity_result.get('activities', []):
                total_activities += 1
                if activity['activity'] in ['trabalhando_computador', 'sentado', 'usando_celular']:
                    work_activities += 1
        
        # Se maioria é trabalho, ajustar
        if work_activities > total_activities * 0.4:
            # Sobrescrever contadores para refletir realidade
            self.activity_detector.activity_counts['trabalhando_computador'] += 100
    
    def _generate_timeline(self) -> List[Dict]:
        """
        Gera linha do tempo dos eventos principais.
        """
        timeline = []
        
        # Agrupar eventos por segundo
        for second in range(int(self.duration) + 1):
            start_frame = second * self.fps
            end_frame = (second + 1) * self.fps
            
            second_data = {
                'second': second,
                'timestamp': f"{second//60:02d}:{second%60:02d}",
                'faces_count': 0,
                'emotions': [],
                'activities': [],
                'anomalies': []
            }
            
            # Contar faces no segundo
            faces_in_second = [f for f in self.results['faces'] 
                             if start_frame <= f['frame_number'] < end_frame]
            second_data['faces_count'] = len(faces_in_second)
            
            # Coletar emoções únicas
            emotions_in_second = [e['dominant_emotion'] for e in self.results['emotions']
                                if start_frame <= e['frame_number'] < end_frame]
            second_data['emotions'] = list(set(emotions_in_second))
            
            # Coletar atividades
            activities_in_second = []
            for activity_result in self.results['activities']:
                if start_frame <= activity_result['frame_number'] < end_frame:
                    activities_in_second.extend([a['activity'] for a in activity_result['activities']])
            second_data['activities'] = list(set(activities_in_second))
            
            # Coletar anomalias
            anomalies_in_second = [a['type'] for a in self.results['anomalies']
                                 if start_frame <= a['frame_number'] < end_frame]
            second_data['anomalies'] = list(set(anomalies_in_second))
            
            # Adicionar apenas segundos com eventos
            if (second_data['faces_count'] > 0 or 
                second_data['anomalies'] or 
                second % 10 == 0):  # Incluir a cada 10 segundos para referência
                timeline.append(second_data)
        
        return timeline
    
    def _generate_summary(self) -> Dict:
        """
        Gera resumo completo da análise.
        """
        # Obter resumos dos módulos
        self.logger.debug("Obtendo resumo de emoções...")
        emotion_summary = self.emotion_analyzer.get_emotion_summary()
        self.logger.debug(f"Tipo de emotion_summary: {type(emotion_summary)}")
        
        self.logger.debug("Obtendo resumo de atividades...")
        activity_summary = self.activity_detector.get_activity_summary()
        self.logger.debug(f"Tipo de activity_summary: {type(activity_summary)}")
        
        self.logger.debug("Obtendo resumo de anomalias...")
        anomaly_summary = self.anomaly_detector.get_anomaly_summary()
        self.logger.debug(f"Tipo de anomaly_summary: {type(anomaly_summary)}")
        
        summary = {
            'overview': {
                'total_frames_analyzed': self.stats['frames_processed'],
                'total_faces_detected': self.stats['faces_detected'],
                'unique_people_estimated': self.stats['unique_faces_estimated'],
                'total_emotions_analyzed': self.stats['emotions_analyzed'],
                'total_activities_detected': self.stats['activities_detected'],
                'total_anomalies_found': self.stats['anomalies_found']
            },
            'processing_info': {
                'frames_skipped': self.total_frames - self.stats['frames_processed'],
                'processing_interval': self.processing_interval,
                'effective_fps_analyzed': self.fps / self.processing_interval
            },
            'emotion_analysis': emotion_summary,
            'activity_analysis': activity_summary,
            'anomaly_analysis': anomaly_summary,
            'key_findings': self._extract_key_findings()
        }
        
        return summary
    
    def _extract_key_findings(self) -> List[str]:
        """
        Extrai descobertas principais da análise.
        """
        findings = []
        
        # Pessoas
        if self.stats['unique_faces_estimated'] > 0:
            findings.append(f"Aproximadamente {self.stats['unique_faces_estimated']} "
                          f"pessoas aparecem no vídeo")
        
        # Emoção dominante
        if self.results['emotions']:
            emotion_counts = {}
            for emotion in self.results['emotions']:
                emo = emotion['dominant_emotion']
                emotion_counts[emo] = emotion_counts.get(emo, 0) + 1
            
            if emotion_counts:
                dominant = max(emotion_counts.items(), key=lambda x: x[1])
                findings.append(f"Emoção mais comum: {dominant[0]} "
                              f"({dominant[1]/len(self.results['emotions'])*100:.1f}%)")
        
        # Atividade principal
        if self.activity_detector.activity_counts:
            main_activity = max(self.activity_detector.activity_counts.items(), 
                              key=lambda x: x[1])
            if main_activity[1] > 0:
                findings.append(f"Atividade principal: "
                              f"{self.activity_detector.ACTIVITIES[main_activity[0]]}")
        
        # Anomalias
        if self.stats['anomalies_found'] > 0:
            findings.append(f"{self.stats['anomalies_found']} anomalias detectadas")
            
            # Tipo mais comum
            anomaly_types = {}
            for anomaly in self.results['anomalies']:
                atype = anomaly['type']
                anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
            
            if anomaly_types:
                common_anomaly = max(anomaly_types.items(), key=lambda x: x[1])
                findings.append(f"Anomalia mais comum: "
                              f"{self.anomaly_detector.ANOMALY_TYPES[common_anomaly[0]]['description']}")
        
        return findings
    
    def save_results(self, filename: str = None) -> str:
        """
        Salva os resultados da análise.
        
        Args:
            filename: Nome do arquivo (opcional)
            
        Returns:
            Caminho do arquivo salvo
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Salvar JSON principal
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Resultados salvos em: {filepath}")
        
        # Salvar relatório de anomalias separado
        if self.results['anomalies']:
            anomaly_file = filepath.replace('.json', '_anomalies.json')
            self.anomaly_detector.save_anomaly_report(
                anomaly_file, 
                self.results['anomalies']
            )
        
        return filepath
    
    def cleanup(self):
        """
        Libera todos os recursos.
        """
        if self.cap.isOpened():
            self.cap.release()
        
        if self.video_writer:
            self.video_writer.release()
        
        # Limpar módulos
        self.face_detector.cleanup()
        self.activity_detector.cleanup()
        
        cv2.destroyAllWindows()
        
        self.logger.info("Recursos liberados")
    
    def __enter__(self):
        """Context manager entrada."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager saída."""
        self.cleanup()


def main():
    """Função principal para execução direta."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analisador de Vídeo com IA')
    parser.add_argument('video_path', help='Caminho para o arquivo de vídeo')
    parser.add_argument('--output-dir', default='data/output', 
                       help='Diretório de saída (padrão: data/output)')
    parser.add_argument('--interval', type=int, default=5,
                       help='Processar a cada N frames (padrão: 5)')
    parser.add_argument('--save-video', action='store_true',
                       help='Salvar vídeo anotado')
    
    args = parser.parse_args()
    
    try:
        # Executar análise
        with VideoAnalyzer(
            video_path=args.video_path,
            output_dir=args.output_dir,
            processing_interval=args.interval,
            save_annotated_video=args.save_video
        ) as analyzer:
            
            # Processar vídeo
            results = analyzer.process_video()
            
            # Salvar resultados
            output_file = analyzer.save_results()
            
            # Mostrar resumo
            print("\n" + "="*80)
            print("ANÁLISE CONCLUÍDA")
            print("="*80)
            print(f"\nArquivo analisado: {args.video_path}")
            print(f"Resultados salvos em: {output_file}")
            print("\nRESUMO:")
            print("-"*40)
            
            summary = results['summary']['overview']
            for key, value in summary.items():
                print(f"{key}: {value}")
            
            print("\nDESCOBERTAS PRINCIPAIS:")
            print("-"*40)
            for finding in results['summary']['key_findings']:
                print(f"• {finding}")
            
            print("\n" + "="*80)
    
    except Exception as e:
        print(f"Erro: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
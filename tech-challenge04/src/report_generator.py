#!/usr/bin/env python3
"""
Report Generator - Módulo para geração de relatórios profissionais
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from collections import Counter
import logging

# Configurar matplotlib para português
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ReportGenerator:
    """
    Gera relatórios profissionais da análise de vídeo.
    
    Formatos suportados:
    - JSON: Dados estruturados completos
    - TXT: Relatório textual formatado
    - HTML: Relatório web interativo
    - PDF: Relatório com gráficos
    """
    
    def __init__(self, results: Dict, output_dir: str = 'data/output'):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            results: Resultados da análise de vídeo
            output_dir: Diretório para salvar relatórios
        """
        self.logger = logging.getLogger(__name__)
        self.results = results
        self.output_dir = output_dir
        
        # Criar diretório se não existir
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp para nomes de arquivo
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_all_reports(self) -> Dict[str, str]:
        """
        Gera todos os tipos de relatório.
        
        Returns:
            Dict com caminhos dos arquivos gerados
        """
        files = {}
        
        # JSON detalhado
        files['json'] = self.generate_json_report()
        
        # Relatório textual
        files['txt'] = self.generate_text_report()
        
        # Relatório HTML
        files['html'] = self.generate_html_report()
        
        # Relatório PDF com gráficos
        files['pdf'] = self.generate_pdf_report()
        
        # CSV com dados tabulares
        files['csv'] = self.generate_csv_data()
        
        self.logger.info(f"Todos os relatórios gerados em: {self.output_dir}")
        
        return files
    
    def generate_json_report(self, filename: str = None) -> str:
        """
        Gera relatório JSON completo.
        """
        if filename is None:
            filename = f"relatorio_completo_{self.timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Estruturar dados para o relatório
        report_data = {
            'relatorio_analise_video': {
                'informacoes_gerais': {
                    'data_processamento': self.results['metadata']['processing_date'],
                    'arquivo_video': self.results['metadata']['video_path'],
                    'duracao_segundos': self.results['metadata']['duration_seconds'],
                    'total_frames': self.results['metadata']['total_frames'],
                    'frames_analisados': self.results['summary']['overview']['total_frames_analyzed'],
                    'fps_original': self.results['metadata']['fps'],
                    'resolucao': f"{self.results['metadata']['width']}x{self.results['metadata']['height']}"
                },
                'resumo_executivo': self.results['summary']['key_findings'],
                'estatisticas': {
                    'pessoas': {
                        'total_deteccoes_rostos': self.results['summary']['overview']['total_faces_detected'],
                        'pessoas_unicas_estimadas': self.results['summary']['overview']['unique_people_estimated']
                    },
                    'emocoes': self._format_emotion_stats(),
                    'atividades': self._format_activity_stats(),
                    'anomalias': self._format_anomaly_stats()
                },
                'linha_tempo': self._format_timeline(),
                'dados_detalhados': {
                    'total_anomalias': len(self.results['anomalies']),
                    'anomalias': self._format_anomalies_detail()
                }
            }
        }
        
        # Salvar JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Relatório JSON salvo: {filepath}")
        return filepath
    
    def generate_text_report(self, filename: str = None) -> str:
        """
        Gera relatório em formato texto.
        """
        if filename is None:
            filename = f"relatorio_analise_{self.timestamp}.txt"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Cabeçalho
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE ANÁLISE DE VÍDEO\n")
            f.write("="*80 + "\n\n")
            
            # Informações do vídeo
            f.write("1. INFORMAÇÕES DO VÍDEO\n")
            f.write("-"*40 + "\n")
            meta = self.results['metadata']
            f.write(f"Arquivo: {meta['video_path']}\n")
            f.write(f"Duração: {meta['duration_seconds']:.2f} segundos\n")
            f.write(f"Resolução: {meta['width']}x{meta['height']}\n")
            f.write(f"FPS: {meta['fps']}\n")
            f.write(f"Total de frames: {meta['total_frames']}\n")
            f.write(f"Data da análise: {meta['processing_date']}\n\n")
            
            # Resumo da análise
            f.write("2. RESUMO DA ANÁLISE\n")
            f.write("-"*40 + "\n")
            overview = self.results['summary']['overview']
            f.write(f"Frames analisados: {overview['total_frames_analyzed']}\n")
            f.write(f"Total de rostos detectados: {overview['total_faces_detected']}\n")
            f.write(f"Pessoas únicas estimadas: {overview['unique_people_estimated']}\n")
            f.write(f"Total de anomalias detectadas: {overview['total_anomalies_found']}\n\n")
            
            # Descobertas principais
            f.write("3. DESCOBERTAS PRINCIPAIS\n")
            f.write("-"*40 + "\n")
            for finding in self.results['summary']['key_findings']:
                f.write(f"• {finding}\n")
            f.write("\n")
            
            # Análise de emoções
            f.write("4. ANÁLISE DE EMOÇÕES\n")
            f.write("-"*40 + "\n")
            emotion_stats = self._format_emotion_stats()
            if emotion_stats:
                for emotion, data in emotion_stats.items():
                    # Verificar se data é um dicionário válido
                    if isinstance(data, dict) and 'count' in data and 'percentage' in data:
                        f.write(f"- {emotion}: {data['count']} ocorrências ({data['percentage']:.1f}%)\n")
                    elif emotion != 'mais_comum':  # Ignorar a entrada 'mais_comum'
                        f.write(f"- {emotion}: dados indisponíveis\n")
                f.write(f"\nEmoção mais comum: {emotion_stats.get('mais_comum', 'N/A')}\n")
            else:
                f.write("Nenhuma emoção detectada\n")
            f.write("\n")
            
            # Análise de atividades
            f.write("5. ATIVIDADES DETECTADAS\n")
            f.write("-"*40 + "\n")
            activity_stats = self._format_activity_stats()
            if activity_stats:
                for activity, data in activity_stats.items():
                    if isinstance(data, dict) and 'count' in data:
                        description = data.get('description', activity)
                        count = data.get('count', 0)
                        f.write(f"- {description}: {count} ocorrências\n")
                    else:
                        f.write(f"- {activity}: dados indisponíveis\n")
            else:
                f.write("Nenhuma atividade detectada\n")
            f.write("\n")
            
            # Anomalias
            f.write("6. ANOMALIAS DETECTADAS\n")
            f.write("-"*40 + "\n")
            f.write(f"Total de anomalias: {len(self.results['anomalies'])}\n\n")
            
            if self.results['anomalies']:
                # Agrupar por tipo
                anomaly_groups = {}
                for anomaly in self.results['anomalies']:
                    atype = anomaly['type']
                    if atype not in anomaly_groups:
                        anomaly_groups[atype] = []
                    anomaly_groups[atype].append(anomaly)
                
                for atype, anomalies in anomaly_groups.items():
                    f.write(f"{atype.upper()} ({len(anomalies)} ocorrências):\n")
                    for i, anomaly in enumerate(anomalies[:5]):  # Primeiras 5
                        f.write(f"  {i+1}. Frame {anomaly['frame_number']} "
                               f"(tempo: {anomaly['timestamp']:.2f}s)\n")
                        f.write(f"     Descrição: {anomaly['description']}\n")
                        f.write(f"     Severidade: {anomaly['severity']}\n")
                    if len(anomalies) > 5:
                        f.write(f"  ... e mais {len(anomalies)-5} ocorrências\n")
                    f.write("\n")
            else:
                f.write("Nenhuma anomalia detectada\n\n")
            
            # Linha do tempo resumida
            f.write("7. LINHA DO TEMPO (PRINCIPAIS EVENTOS)\n")
            f.write("-"*40 + "\n")
            timeline = self.results.get('timeline', [])
            for event in timeline[:20]:  # Primeiros 20 eventos
                if event['anomalies'] or event['faces_count'] > 2:
                    f.write(f"Tempo {event['timestamp']}: ")
                    if event['faces_count']:
                        f.write(f"{event['faces_count']} pessoas | ")
                    if event['emotions']:
                        f.write(f"Emoções: {', '.join(event['emotions'])} | ")
                    if event['anomalies']:
                        f.write(f"ANOMALIA: {', '.join(event['anomalies'])}")
                    f.write("\n")
            
            # Rodapé
            f.write("\n" + "="*80 + "\n")
            f.write("FIM DO RELATÓRIO\n")
            f.write("="*80 + "\n")
        
        self.logger.info(f"Relatório TXT salvo: {filepath}")
        return filepath
    
    def generate_html_report(self, filename: str = None) -> str:
        """
        Gera relatório HTML interativo.
        """
        if filename is None:
            filename = f"relatorio_web_{self.timestamp}.html"
        
        filepath = os.path.join(self.output_dir, filename)
        
        html_content = """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório de Análise de Vídeo</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 30px;
        }}
        .metric-box {{
            display: inline-block;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 20px;
            margin: 10px;
            text-align: center;
            min-width: 150px;
        }}
        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }}
        .anomaly-item {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
        }}
        .anomaly-high {{
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }}
        .timeline-item {{
            border-left: 3px solid #007bff;
            padding-left: 20px;
            margin-left: 10px;
            margin-bottom: 20px;
        }}
        .chart-container {{
            margin: 20px 0;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #007bff;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .key-finding {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            color: #155724;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Relatório de Análise de Vídeo</h1>
        
        <div style="text-align: center; margin-bottom: 30px;">
            <p><strong>Data da Análise:</strong> {processing_date}</p>
            <p><strong>Arquivo:</strong> {video_path}</p>
        </div>
        
        <h2>📈 Métricas Principais</h2>
        <div style="text-align: center;">
            <div class="metric-box">
                <div class="metric-value">{total_faces}</div>
                <div class="metric-label">Rostos Detectados</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{unique_people}</div>
                <div class="metric-label">Pessoas Estimadas</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{total_anomalies}</div>
                <div class="metric-label">Anomalias</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{duration:.1f}s</div>
                <div class="metric-label">Duração</div>
            </div>
        </div>
        
        <h2>🎯 Descobertas Principais</h2>
        {key_findings_html}
        
        <h2>😊 Distribuição de Emoções</h2>
        <table>
            <tr>
                <th>Emoção</th>
                <th>Ocorrências</th>
                <th>Percentual</th>
            </tr>
            {emotion_table_rows}
        </table>
        
        <h2>🏃 Atividades Detectadas</h2>
        <table>
            <tr>
                <th>Atividade</th>
                <th>Ocorrências</th>
                <th>Descrição</th>
            </tr>
            {activity_table_rows}
        </table>
        
        <h2>⚠️ Anomalias Detectadas</h2>
        {anomalies_html}
        
        <h2>⏱️ Linha do Tempo</h2>
        <div style="max-height: 400px; overflow-y: auto;">
            {timeline_html}
        </div>
        
        <hr style="margin-top: 50px;">
        <p style="text-align: center; color: #6c757d;">
            Relatório gerado automaticamente pelo Sistema de Análise de Vídeo com IA
        </p>
    </div>
</body>
</html>
        """
        
        # Preencher template com dados
        processing_date = datetime.fromisoformat(
            self.results['metadata']['processing_date']
        ).strftime("%d/%m/%Y %H:%M")
        
        # Descobertas principais
        key_findings_html = ""
        for finding in self.results['summary']['key_findings']:
            key_findings_html += f'<div class="key-finding">✓ {finding}</div>'
        
        # Tabela de emoções
        emotion_rows = ""
        emotion_stats = self._format_emotion_stats()
        for emotion, data in emotion_stats.items():
            if isinstance(data, dict) and 'count' in data and 'percentage' in data and emotion != 'mais_comum':
                emotion_rows += f"""
                <tr>
                    <td>{emotion}</td>
                    <td>{data['count']}</td>
                    <td>{data['percentage']:.1f}%</td>
                </tr>
                """
        
        # Se não há dados, adicionar linha indicativa
        if not emotion_rows:
            emotion_rows = """
                <tr>
                    <td colspan="3">Nenhuma emoção detectada</td>
                </tr>
                """
        
        # Tabela de atividades
        activity_rows = ""
        activity_stats = self._format_activity_stats()
        for activity, data in activity_stats.items():
            if isinstance(data, dict) and 'count' in data:
                activity_rows += f"""
                <tr>
                    <td>{activity}</td>
                    <td>{data['count']}</td>
                    <td>{data.get('description', '')}</td>
                </tr>
                """
        
        # Se não há dados, adicionar linha indicativa
        if not activity_rows:
            activity_rows = """
                <tr>
                    <td colspan="3">Nenhuma atividade detectada</td>
                </tr>
                """
        
        # Anomalias
        anomalies_html = ""
        if self.results['anomalies']:
            for anomaly in self.results['anomalies'][:10]:  # Top 10
                severity_class = 'anomaly-high' if anomaly['severity'] == 'alta' else 'anomaly-item'
                anomalies_html += f"""
                <div class="{severity_class}">
                    <strong>{anomaly['type']}</strong> - Frame {anomaly['frame_number']} 
                    ({anomaly['timestamp']:.2f}s)<br>
                    {anomaly['description']}<br>
                    <small>Severidade: {anomaly['severity']}</small>
                </div>
                """
        else:
            anomalies_html = "<p>Nenhuma anomalia detectada</p>"
        
        # Timeline
        timeline_html = ""
        for event in self.results.get('timeline', [])[:30]:
            if event['anomalies'] or event['faces_count'] > 0:
                emotions_text = ""
                if event['emotions']:
                    emotions_text = f"Emoções: {', '.join(event['emotions'])}<br>"
                
                anomalies_text = ""
                if event['anomalies']:
                    anomalies_text = f"<span style='color: red;'>Anomalia: {', '.join(event['anomalies'])}</span>"
                
                timeline_html += f"""
                <div class="timeline-item">
                    <strong>{event['timestamp']}</strong><br>
                    Pessoas: {event['faces_count']}<br>
                    {emotions_text}
                    {anomalies_text}
                </div>
                """
        
        # Substituir placeholders
        html_content = html_content.format(
            processing_date=processing_date,
            video_path=os.path.basename(self.results['metadata']['video_path']),
            total_faces=self.results['summary']['overview']['total_faces_detected'],
            unique_people=self.results['summary']['overview']['unique_people_estimated'],
            total_anomalies=self.results['summary']['overview']['total_anomalies_found'],
            duration=self.results['metadata']['duration_seconds'],
            key_findings_html=key_findings_html,
            emotion_table_rows=emotion_rows,
            activity_table_rows=activity_rows,
            anomalies_html=anomalies_html,
            timeline_html=timeline_html
        )
        
        # Salvar HTML
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Relatório HTML salvo: {filepath}")
        return filepath
    
    def generate_pdf_report(self, filename: str = None) -> str:
        """
        Gera relatório PDF com gráficos.
        """
        if filename is None:
            filename = f"relatorio_visual_{self.timestamp}.pdf"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with PdfPages(filepath) as pdf:
            # Página 1: Resumo
            self._create_summary_page(pdf)
            
            # Página 2: Gráficos de emoções
            self._create_emotion_charts(pdf)
            
            # Página 3: Gráficos de atividades
            self._create_activity_charts(pdf)
            
            # Página 4: Análise temporal
            self._create_temporal_analysis(pdf)
            
            # Página 5: Anomalias
            self._create_anomaly_analysis(pdf)
            
            # Metadata do PDF
            d = pdf.infodict()
            d['Title'] = 'Relatório de Análise de Vídeo'
            d['Author'] = 'Sistema de Análise com IA'
            d['Subject'] = 'Análise automatizada de vídeo'
            d['Keywords'] = 'Video Analysis, AI, Computer Vision'
            d['CreationDate'] = datetime.now()
        
        self.logger.info(f"Relatório PDF salvo: {filepath}")
        return filepath
    
    def _create_summary_page(self, pdf):
        """Cria página de resumo para o PDF."""
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle('Relatório de Análise de Vídeo', fontsize=20, fontweight='bold')
        
        # Texto do resumo
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_text = f"""
        DATA DA ANÁLISE: {datetime.now().strftime('%d/%m/%Y %H:%M')}
        
        ARQUIVO: {os.path.basename(self.results['metadata']['video_path'])}
        
        INFORMAÇÕES DO VÍDEO:
        • Duração: {self.results['metadata']['duration_seconds']:.2f} segundos
        • Resolução: {self.results['metadata']['width']}x{self.results['metadata']['height']}
        • FPS: {self.results['metadata']['fps']}
        • Total de frames: {self.results['metadata']['total_frames']}
        
        RESULTADOS DA ANÁLISE:
        • Frames analisados: {self.results['summary']['overview']['total_frames_analyzed']}
        • Total de rostos detectados: {self.results['summary']['overview']['total_faces_detected']}
        • Pessoas únicas estimadas: {self.results['summary']['overview']['unique_people_estimated']}
        • Total de anomalias: {self.results['summary']['overview']['total_anomalies_found']}
        
        DESCOBERTAS PRINCIPAIS:
        """
        
        for finding in self.results['summary']['key_findings']:
            summary_text += f"\n• {finding}"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_emotion_charts(self, pdf):
        """Cria gráficos de análise de emoções."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle('Análise de Emoções', fontsize=16, fontweight='bold')
        
        # Preparar dados
        emotion_data = self._prepare_emotion_data()
        
        if emotion_data:
            # Gráfico de pizza
            ax1.pie(emotion_data['counts'], labels=emotion_data['labels'], 
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('Distribuição de Emoções')
            
            # Gráfico de barras
            ax2.bar(emotion_data['labels'], emotion_data['counts'])
            ax2.set_title('Contagem de Emoções')
            ax2.set_xlabel('Emoção')
            ax2.set_ylabel('Ocorrências')
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Timeline de emoções
            self._plot_emotion_timeline(ax3)
            
            # Heatmap de emoções por tempo
            self._plot_emotion_heatmap(ax4)
        else:
            for ax in [ax1, ax2, ax3, ax4]:
                ax.text(0.5, 0.5, 'Sem dados de emoções', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_activity_charts(self, pdf):
        """Cria gráficos de análise de atividades."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle('Análise de Atividades', fontsize=16, fontweight='bold')
        
        # Preparar dados
        activity_data = self._prepare_activity_data()
        
        if activity_data:
            # Gráfico de barras horizontais
            ax1.barh(activity_data['labels'], activity_data['counts'])
            ax1.set_title('Atividades Detectadas')
            ax1.set_xlabel('Ocorrências')
            
            # Timeline de atividades
            self._plot_activity_timeline(ax2)
        else:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'Sem dados de atividades', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_temporal_analysis(self, pdf):
        """Cria análise temporal."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5))
        fig.suptitle('Análise Temporal', fontsize=16, fontweight='bold')
        
        # Pessoas detectadas ao longo do tempo
        timeline_data = self._prepare_timeline_data()
        
        if timeline_data:
            ax1.plot(timeline_data['seconds'], timeline_data['faces_count'], 
                    marker='o', markersize=4)
            ax1.set_title('Pessoas Detectadas ao Longo do Tempo')
            ax1.set_xlabel('Tempo (s)')
            ax1.set_ylabel('Número de Pessoas')
            ax1.grid(True, alpha=0.3)
            
            # Anomalias ao longo do tempo
            anomaly_times = [a['timestamp'] for a in self.results['anomalies']]
            if anomaly_times:
                ax2.hist(anomaly_times, bins=20, color='red', alpha=0.7)
                ax2.set_title('Distribuição Temporal de Anomalias')
                ax2.set_xlabel('Tempo (s)')
                ax2.set_ylabel('Número de Anomalias')
                ax2.grid(True, alpha=0.3)
            else:
                ax2.text(0.5, 0.5, 'Sem anomalias detectadas', 
                        transform=ax2.transAxes, ha='center', va='center')
                ax2.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def _create_anomaly_analysis(self, pdf):
        """Cria análise de anomalias."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle('Análise de Anomalias', fontsize=16, fontweight='bold')
        
        if self.results['anomalies']:
            # Tipos de anomalias
            anomaly_types = [a['type'] for a in self.results['anomalies']]
            type_counts = Counter(anomaly_types)
            
            ax1.bar(type_counts.keys(), type_counts.values(), color='red', alpha=0.7)
            ax1.set_title('Tipos de Anomalias')
            ax1.set_xlabel('Tipo')
            ax1.set_ylabel('Ocorrências')
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Severidade das anomalias
            severities = [a['severity'] for a in self.results['anomalies']]
            severity_counts = Counter(severities)
            
            colors = {'baixa': 'yellow', 'média': 'orange', 'alta': 'red'}
            ax2.pie(severity_counts.values(), labels=severity_counts.keys(),
                   autopct='%1.1f%%', colors=[colors.get(s, 'gray') for s in severity_counts.keys()])
            ax2.set_title('Distribuição por Severidade')
        else:
            for ax in [ax1, ax2]:
                ax.text(0.5, 0.5, 'Sem anomalias detectadas', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.axis('off')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def generate_csv_data(self, filename: str = None) -> str:
        """
        Gera dados em formato CSV para análise posterior.
        """
        if filename is None:
            filename = f"dados_analise_{self.timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Preparar dados para DataFrame
        data = []
        
        # Adicionar dados de cada frame analisado
        for frame_summary in self.results.get('frame_summaries', []):
            row = {
                'frame_number': frame_summary['frame_number'],
                'timestamp': frame_summary['timestamp'],
                'faces_count': frame_summary['faces_count'],
                'emotions': ', '.join([e['emotion'] for e in frame_summary['emotions']]),
                'activities': ', '.join([a['activity'] for a in frame_summary['activities']]),
                'anomaly_count': len(frame_summary['anomalies']),
                'anomaly_types': ', '.join([a['type'] for a in frame_summary['anomalies']])
            }
            data.append(row)
        
        # Criar DataFrame e salvar
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        self.logger.info(f"Dados CSV salvos: {filepath}")
        return filepath
    
    # Métodos auxiliares para formatação de dados
    
    def _format_emotion_stats(self) -> Dict:
        """Formata estatísticas de emoções."""
        try:
            emotion_summary = self.results['summary'].get('emotion_analysis', {})
            
            # Verificar se existe emotion_distribution e se é um dict
            if not isinstance(emotion_summary, dict) or 'emotion_distribution' not in emotion_summary:
                return {}
            
            emotion_distribution = emotion_summary['emotion_distribution']
            if not isinstance(emotion_distribution, dict):
                return {}
            
            stats = {}
            for emotion, data in emotion_distribution.items():
                # Verificar se data é um dicionário e tem as chaves necessárias
                if isinstance(data, dict) and 'count' in data and 'percentage' in data:
                    translation = data.get('translation', emotion)
                    stats[translation] = {
                        'count': data['count'],
                        'percentage': data['percentage']
                    }
                else:
                    # Se data não é um dict, criar entrada padrão
                    stats[emotion] = {
                        'count': 0,
                        'percentage': 0.0
                    }
            
            # Adicionar mais comum se existir e for válido
            if ('most_common_emotion' in emotion_summary and 
                isinstance(emotion_summary['most_common_emotion'], dict) and
                'translation' in emotion_summary['most_common_emotion']):
                stats['mais_comum'] = emotion_summary['most_common_emotion']['translation']
            
            return stats
            
        except Exception as e:
            self.logger.warning(f"Erro ao formatar estatísticas de emoções: {str(e)}")
            return {}
    
    def _format_activity_stats(self) -> Dict:
        """Formata estatísticas de atividades."""
        try:
            activity_summary = self.results['summary'].get('activity_analysis', {})
            
            # Verificar se existe activity_distribution e se é um dict
            if not isinstance(activity_summary, dict) or 'activity_distribution' not in activity_summary:
                return {}
            
            activity_distribution = activity_summary['activity_distribution']
            if not isinstance(activity_distribution, dict):
                return {}
            
            # Validar que cada entrada é um dicionário válido
            validated_stats = {}
            for activity, data in activity_distribution.items():
                if isinstance(data, dict):
                    validated_stats[activity] = data
                else:
                    # Criar entrada padrão se data não é um dict
                    validated_stats[activity] = {
                        'count': 0,
                        'percentage': 0.0,
                        'description': activity
                    }
            
            return validated_stats
            
        except Exception as e:
            self.logger.warning(f"Erro ao formatar estatísticas de atividades: {str(e)}")
            return {}
    
    def _format_anomaly_stats(self) -> Dict:
        """Formata estatísticas de anomalias."""
        try:
            anomaly_summary = self.results['summary'].get('anomaly_analysis', {})
            
            # Verificar se anomaly_summary é um dict válido
            if not isinstance(anomaly_summary, dict):
                return {
                    'total': 0,
                    'taxa_por_frame': 0.0,
                    'distribuicao': {}
                }
            
            return {
                'total': anomaly_summary.get('total_anomalies', 0),
                'taxa_por_frame': anomaly_summary.get('anomaly_rate', 0.0),
                'distribuicao': anomaly_summary.get('anomaly_distribution', {})
            }
            
        except Exception as e:
            self.logger.warning(f"Erro ao formatar estatísticas de anomalias: {str(e)}")
            return {
                'total': 0,
                'taxa_por_frame': 0.0,
                'distribuicao': {}
            }
    
    def _format_timeline(self) -> List[Dict]:
        """Formata linha do tempo para o relatório."""
        timeline = []
        for event in self.results.get('timeline', []):
            if event['anomalies'] or event['faces_count'] > 0:
                timeline.append({
                    'tempo': event['timestamp'],
                    'segundo': event['second'],
                    'pessoas': event['faces_count'],
                    'emocoes': event['emotions'],
                    'atividades': event['activities'],
                    'anomalias': event['anomalies']
                })
        return timeline
    
    def _format_anomalies_detail(self) -> List[Dict]:
        """Formata detalhes das anomalias."""
        anomalies = []
        for anomaly in self.results['anomalies']:
            anomalies.append({
                'tipo': anomaly['type'],
                'frame': anomaly['frame_number'],
                'tempo_segundos': anomaly['timestamp'],
                'descricao': anomaly['description'],
                'severidade': anomaly['severity'],
                'confianca': anomaly.get('confidence', 0)
            })
        return anomalies
    
    def _prepare_emotion_data(self) -> Dict:
        """Prepara dados de emoções para gráficos."""
        emotion_stats = self._format_emotion_stats()
        if emotion_stats and 'mais_comum' in emotion_stats:
            # Remover 'mais_comum' que não é uma emoção
            emotion_stats.pop('mais_comum', None)
            
            labels = list(emotion_stats.keys())
            counts = [data['count'] for data in emotion_stats.values()]
            
            return {
                'labels': labels,
                'counts': counts
            }
        return {}
    
    def _prepare_activity_data(self) -> Dict:
        """Prepara dados de atividades para gráficos."""
        activity_stats = self._format_activity_stats()
        if activity_stats:
            labels = []
            counts = []
            
            for activity, data in activity_stats.items():
                if isinstance(data, dict) and 'count' in data:
                    labels.append(data.get('description', activity))
                    counts.append(data['count'])
            
            return {
                'labels': labels,
                'counts': counts
            }
        return {}
    
    def _prepare_timeline_data(self) -> Dict:
        """Prepara dados da timeline para gráficos."""
        timeline = self.results.get('timeline', [])
        if timeline:
            seconds = [event['second'] for event in timeline]
            faces_count = [event['faces_count'] for event in timeline]
            
            return {
                'seconds': seconds,
                'faces_count': faces_count
            }
        return {}
    
    def _plot_emotion_timeline(self, ax):
        """Plota timeline de emoções."""
        # Simplificado - em produção seria mais elaborado
        ax.set_title('Timeline de Emoções')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Emoção')
        
        # Coletar dados de emoções ao longo do tempo
        emotion_times = {}
        for emotion in self.results.get('emotions', []):
            emo_type = emotion['dominant_emotion']
            timestamp = emotion['frame_number'] / self.results['metadata']['fps']
            
            if emo_type not in emotion_times:
                emotion_times[emo_type] = []
            emotion_times[emo_type].append(timestamp)
        
        # Plotar
        y_pos = 0
        for emotion, times in emotion_times.items():
            ax.scatter(times, [y_pos] * len(times), label=emotion, s=20)
            y_pos += 1
        
        ax.set_yticks(range(len(emotion_times)))
        ax.set_yticklabels(list(emotion_times.keys()))
        ax.grid(True, alpha=0.3)
    
    def _plot_emotion_heatmap(self, ax):
        """Plota heatmap de emoções."""
        ax.set_title('Intensidade de Emoções por Tempo')
        ax.text(0.5, 0.5, 'Heatmap de emoções\n(implementação futura)', 
               transform=ax.transAxes, ha='center', va='center')
    
    def _plot_activity_timeline(self, ax):
        """Plota timeline de atividades."""
        ax.set_title('Timeline de Atividades')
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Atividade')
        
        # Coletar dados
        activity_times = {}
        for activity_result in self.results.get('activities', []):
            timestamp = activity_result['timestamp']
            for activity in activity_result['activities']:
                act_type = activity['activity']
                if act_type not in activity_times:
                    activity_times[act_type] = []
                activity_times[act_type].append(timestamp)
        
        # Plotar
        y_pos = 0
        for activity, times in activity_times.items():
            ax.scatter(times, [y_pos] * len(times), label=activity, s=20)
            y_pos += 1
        
        if activity_times:
            ax.set_yticks(range(len(activity_times)))
            ax.set_yticklabels(list(activity_times.keys()))
        ax.grid(True, alpha=0.3)


# Função de teste
def test_report_generator():
    """Testa o gerador de relatórios com dados de exemplo."""
    
    # Criar dados de exemplo
    sample_results = {
        'metadata': {
            'video_path': 'test_video.mp4',
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'total_frames': 3000,
            'duration_seconds': 100.0,
            'processing_date': datetime.now().isoformat()
        },
        'summary': {
            'overview': {
                'total_frames_analyzed': 600,
                'total_faces_detected': 150,
                'unique_people_estimated': 5,
                'total_emotions_analyzed': 150,
                'total_activities_detected': 50,
                'total_anomalies_found': 10
            },
            'key_findings': [
                'Aproximadamente 5 pessoas aparecem no vídeo',
                'Emoção mais comum: feliz (45%)',
                'Atividade principal: conversando',
                '10 anomalias detectadas'
            ],
            'emotion_analysis': {
                'emotion_distribution': {
                    'happy': {'count': 68, 'percentage': 45.3, 'translation': 'feliz'},
                    'neutral': {'count': 45, 'percentage': 30.0, 'translation': 'neutro'},
                    'sad': {'count': 22, 'percentage': 14.7, 'translation': 'triste'},
                    'surprise': {'count': 15, 'percentage': 10.0, 'translation': 'surpresa'}
                },
                'most_common_emotion': {
                    'emotion': 'happy',
                    'translation': 'feliz',
                    'count': 68
                }
            },
            'activity_analysis': {
                'activity_distribution': {
                    'conversando': {'count': 30, 'percentage': 60, 'description': 'Conversando'},
                    'sentado': {'count': 15, 'percentage': 30, 'description': 'Sentado'},
                    'caminhando': {'count': 5, 'percentage': 10, 'description': 'Caminhando'}
                }
            },
            'anomaly_analysis': {
                'total_anomalies': 10,
                'anomaly_rate': 0.017,
                'anomaly_distribution': {
                    'careta_exagerada': {'count': 6, 'percentage': 60},
                    'elemento_grafico': {'count': 3, 'percentage': 30},
                    'movimento_brusco': {'count': 1, 'percentage': 10}
                }
            }
        },
        'faces': [],
        'emotions': [],
        'activities': [],
        'anomalies': [
            {
                'type': 'careta_exagerada',
                'frame_number': 150,
                'timestamp': 5.0,
                'description': 'Expressão surprise com intensidade 92%',
                'severity': 'alta',
                'confidence': 0.92
            }
        ],
        'timeline': [
            {
                'second': 0,
                'timestamp': '00:00',
                'faces_count': 2,
                'emotions': ['happy', 'neutral'],
                'activities': ['conversando'],
                'anomalies': []
            },
            {
                'second': 5,
                'timestamp': '00:05',
                'faces_count': 3,
                'emotions': ['happy', 'surprise'],
                'activities': ['conversando'],
                'anomalies': ['careta_exagerada']
            }
        ],
        'frame_summaries': []
    }
    
    # Criar gerador
    generator = ReportGenerator(sample_results, 'test_reports')
    
    # Gerar todos os relatórios
    print("Gerando relatórios de teste...")
    files = generator.generate_all_reports()
    
    print("\nRelatórios gerados:")
    for format_type, filepath in files.items():
        print(f"- {format_type.upper()}: {filepath}")
    
    print("\nTeste concluído!")


if __name__ == "__main__":
    test_report_generator()
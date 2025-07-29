#!/usr/bin/env python3
"""
Tech Challenge - Sistema de Análise de Vídeo com IA
Script principal para execução da análise completa
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json
import time
from pathlib import Path

# Adicionar diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Importar módulos do projeto
from video_analyzer import VideoAnalyzer
from report_generator import ReportGenerator

# Configurar logging
def setup_logging(log_level='INFO'):
    """Configura o sistema de logging."""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def print_banner():
    """Exibe banner do sistema."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║          🎥 SISTEMA DE ANÁLISE DE VÍDEO COM IA 🤖           ║
    ║                                                              ║
    ║                    Tech Challenge - FIAP                     ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def validate_video_file(video_path):
    """Valida se o arquivo de vídeo existe e é válido."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {video_path}")
    
    # Verificar extensão
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in valid_extensions:
        raise ValueError(f"Formato de vídeo não suportado: {ext}")
    
    # Verificar tamanho
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise ValueError("O arquivo de vídeo está vazio")
    
    return True

def create_output_structure(base_dir='data/output'):
    """Cria estrutura de diretórios para saída."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f'analysis_{timestamp}')
    
    # Criar subdiretórios
    dirs = [
        output_dir,
        os.path.join(output_dir, 'reports'),
        os.path.join(output_dir, 'visualizations'),
        os.path.join(output_dir, 'data')
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return output_dir

def run_analysis(video_path, output_dir, processing_interval=5, save_video=False):
    """
    Executa a análise completa do vídeo.
    
    Args:
        video_path: Caminho para o vídeo
        output_dir: Diretório de saída
        processing_interval: Intervalo de processamento
        save_video: Se deve salvar vídeo anotado
    
    Returns:
        Dict com resultados e caminhos dos arquivos gerados
    """
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("INICIANDO ANÁLISE DE VÍDEO")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # 1. Criar analisador e processar vídeo
        logger.info(f"Analisando vídeo: {video_path}")
        
        with VideoAnalyzer(
            video_path=video_path,
            output_dir=os.path.join(output_dir, 'data'),
            processing_interval=processing_interval,
            save_annotated_video=save_video
        ) as analyzer:
            
            # Processar vídeo
            results = analyzer.process_video()
            
            # Salvar resultados brutos
            results_file = analyzer.save_results()
            logger.info(f"Resultados salvos: {results_file}")
        
        # 2. Gerar relatórios
        logger.info("Gerando relatórios...")
        report_gen = ReportGenerator(
            results=results,
            output_dir=os.path.join(output_dir, 'reports')
        )
        
        report_files = report_gen.generate_all_reports()
        
        # 3. Criar resumo da análise
        analysis_time = time.time() - start_time
        
        summary = {
            'status': 'success',
            'video_file': video_path,
            'output_directory': output_dir,
            'processing_time_seconds': round(analysis_time, 2),
            'processing_time_formatted': format_time(analysis_time),
            'results_summary': {
                'total_frames': results['metadata']['total_frames'],
                'frames_analyzed': results['summary']['overview']['total_frames_analyzed'],
                'faces_detected': results['summary']['overview']['total_faces_detected'],
                'unique_people': results['summary']['overview']['unique_people_estimated'],
                'total_anomalies': results['summary']['overview']['total_anomalies_found']
            },
            'generated_files': {
                'raw_results': results_file,
                'reports': report_files
            },
            'key_findings': results['summary']['key_findings']
        }
        
        # Salvar resumo
        summary_file = os.path.join(output_dir, 'analysis_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info("="*60)
        logger.info("ANÁLISE CONCLUÍDA COM SUCESSO")
        logger.info(f"Tempo total: {format_time(analysis_time)}")
        logger.info(f"Resultados em: {output_dir}")
        logger.info("="*60)
        
        return summary
        
    except Exception as e:
        logger.error(f"Erro durante análise: {str(e)}")
        logger.exception("Detalhes do erro:")
        
        # Criar resumo de erro
        summary = {
            'status': 'error',
            'video_file': video_path,
            'error_message': str(e),
            'error_type': type(e).__name__,
            'processing_time_seconds': time.time() - start_time
        }
        
        # Salvar log de erro
        error_file = os.path.join(output_dir, 'error_log.json')
        with open(error_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        raise

def format_time(seconds):
    """Formata tempo em formato legível."""
    if seconds < 60:
        return f"{seconds:.1f} segundos"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutos"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} horas"

def print_results_summary(summary):
    """Imprime resumo dos resultados."""
    print("\n" + "="*60)
    print("📊 RESUMO DA ANÁLISE")
    print("="*60)
    
    if summary['status'] == 'success':
        print(f"\n✅ Status: SUCESSO")
        print(f"⏱️  Tempo de processamento: {summary['processing_time_formatted']}")
        print(f"📁 Resultados salvos em: {summary['output_directory']}")
        
        print("\n📈 ESTATÍSTICAS:")
        stats = summary['results_summary']
        print(f"   • Frames totais: {stats['total_frames']}")
        print(f"   • Frames analisados: {stats['frames_analyzed']}")
        print(f"   • Rostos detectados: {stats['faces_detected']}")
        print(f"   • Pessoas estimadas: {stats['unique_people']}")
        print(f"   • Anomalias encontradas: {stats['total_anomalies']}")
        
        print("\n🎯 DESCOBERTAS PRINCIPAIS:")
        for finding in summary['key_findings']:
            print(f"   • {finding}")
        
        print("\n📄 ARQUIVOS GERADOS:")
        print(f"   • Dados brutos: {os.path.basename(summary['generated_files']['raw_results'])}")
        for format_type, filepath in summary['generated_files']['reports'].items():
            print(f"   • Relatório {format_type.upper()}: {os.path.basename(filepath)}")
    
    else:
        print(f"\n❌ Status: ERRO")
        print(f"   Erro: {summary['error_message']}")
    
    print("\n" + "="*60)

def main():
    """Função principal."""
    # Parser de argumentos
    parser = argparse.ArgumentParser(
        description='Sistema de Análise de Vídeo com IA',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python main.py video.mp4
  python main.py video.mp4 --interval 10 --save-video
  python main.py video.mp4 --output-dir ./resultados
        """
    )
    
    parser.add_argument(
        'video_path',
        help='Caminho para o arquivo de vídeo a ser analisado'
    )
    
    parser.add_argument(
        '--output-dir',
        default='data/output',
        help='Diretório base para salvar resultados (padrão: data/output)'
    )
    
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Processar a cada N frames (padrão: 5)'
    )
    
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Salvar vídeo anotado com detecções'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Nível de log (padrão: INFO)'
    )
    
    parser.add_argument(
        '--no-reports',
        action='store_true',
        help='Pular geração de relatórios (apenas salvar dados brutos)'
    )
    
    args = parser.parse_args()
    
    # Exibir banner
    print_banner()
    
    # Configurar logging
    logger = setup_logging(args.log_level)
    
    try:
        # Validar vídeo
        print(f"\n🔍 Validando arquivo de vídeo...")
        validate_video_file(args.video_path)
        print(f"✅ Arquivo válido: {args.video_path}")
        
        # Criar estrutura de saída
        output_dir = create_output_structure(args.output_dir)
        print(f"📁 Diretório de saída: {output_dir}")
        
        # Executar análise
        print(f"\n🚀 Iniciando análise...")
        print(f"   • Intervalo de processamento: 1 a cada {args.interval} frames")
        print(f"   • Salvar vídeo anotado: {'Sim' if args.save_video else 'Não'}")
        print(f"\n⏳ Isso pode demorar alguns minutos...\n")
        
        summary = run_analysis(
            video_path=args.video_path,
            output_dir=output_dir,
            processing_interval=args.interval,
            save_video=args.save_video
        )
        
        # Exibir resumo
        print_results_summary(summary)
        
        # Abrir diretório de resultados (Windows)
        if sys.platform == 'win32':
            os.startfile(output_dir)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Análise interrompida pelo usuário")
        return 1
        
    except Exception as e:
        logger.error(f"\n\n❌ Erro fatal: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
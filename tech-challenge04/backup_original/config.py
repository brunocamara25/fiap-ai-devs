#!/usr/bin/env python3
"""
Configurações do Sistema de Análise de Vídeo
"""

# Configurações de processamento
PROCESSING_CONFIGS = {
    'conservative': {
        'processing_interval': 3,
        'emotion_threshold': 0.98,
        'movement_threshold': 0.7,
        'graphic_threshold': 0.9,
        'min_anomaly_persistence': 3,
        'description': 'Configuração conservadora - poucos falsos positivos'
    },
    'normal': {
        'processing_interval': 2,
        'emotion_threshold': 0.95,
        'movement_threshold': 0.6,
        'graphic_threshold': 0.8,
        'min_anomaly_persistence': 2,
        'description': 'Configuração padrão - equilíbrio entre precisão e recall'
    },
    'sensitive': {
        'processing_interval': 1,
        'emotion_threshold': 0.85,
        'movement_threshold': 0.4,
        'graphic_threshold': 0.7,
        'min_anomaly_persistence': 1,
        'description': 'Configuração sensível - detecta mais anomalias'
    }
}

# Configuração padrão
DEFAULT_CONFIG = 'normal'

def get_config(config_name: str = None) -> dict:
    """
    Retorna a configuração especificada.
    
    Args:
        config_name: Nome da configuração ('conservative', 'normal', 'sensitive')
        
    Returns:
        Dict com parâmetros de configuração
    """
    if config_name is None:
        config_name = DEFAULT_CONFIG
    
    if config_name not in PROCESSING_CONFIGS:
        raise ValueError(f"Configuração '{config_name}' não encontrada. "
                        f"Opções: {list(PROCESSING_CONFIGS.keys())}")
    
    return PROCESSING_CONFIGS[config_name].copy()

def list_configs() -> dict:
    """
    Lista todas as configurações disponíveis.
    
    Returns:
        Dict com todas as configurações e suas descrições
    """
    return {name: config['description'] 
            for name, config in PROCESSING_CONFIGS.items()}

# Configurações específicas por tipo de vídeo
VIDEO_TYPE_CONFIGS = {
    'meeting': {
        'base_config': 'conservative',
        'suspicious_emotions': ['fear', 'disgust'],
        'focus_on_movement': False,
        'description': 'Reuniões e apresentações'
    },
    'classroom': {
        'base_config': 'normal',
        'suspicious_emotions': ['fear'],
        'focus_on_movement': True,
        'description': 'Sala de aula e eventos educativos'
    },
    'social': {
        'base_config': 'conservative',
        'suspicious_emotions': ['fear', 'disgust'],
        'focus_on_movement': False,
        'description': 'Eventos sociais e festividades'
    },
    'security': {
        'base_config': 'sensitive',
        'suspicious_emotions': ['fear', 'disgust', 'surprise'],
        'focus_on_movement': True,
        'description': 'Análise de segurança'
    }
}

def get_video_config(video_type: str, base_config: str = None) -> dict:
    """
    Retorna configuração otimizada para tipo de vídeo.
    
    Args:
        video_type: Tipo do vídeo ('meeting', 'classroom', 'social', 'security')
        base_config: Configuração base a usar (opcional)
        
    Returns:
        Dict com configuração otimizada
    """
    if video_type not in VIDEO_TYPE_CONFIGS:
        raise ValueError(f"Tipo de vídeo '{video_type}' não suportado. "
                        f"Opções: {list(VIDEO_TYPE_CONFIGS.keys())}")
    
    video_config = VIDEO_TYPE_CONFIGS[video_type]
    
    # Usar configuração base especificada ou a padrão do tipo
    if base_config is None:
        base_config = video_config['base_config']
    
    # Obter configuração base
    config = get_config(base_config)
    
    # Aplicar modificações específicas do tipo de vídeo
    config.update({
        'video_type': video_type,
        'suspicious_emotions': video_config['suspicious_emotions'],
        'focus_on_movement': video_config['focus_on_movement']
    })
    
    return config 
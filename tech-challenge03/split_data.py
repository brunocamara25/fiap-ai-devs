#!/usr/bin/env python3
"""
Script para dividir dados de fine-tuning em conjuntos de treinamento e validação.
"""

import argparse
import json
import random
from typing import Any, Dict, List


def split_fine_tuning_data(input_file: str, train_ratio: float = 0.8, shuffle: bool = True):
    """
    Divide dados de fine-tuning em conjuntos de treinamento e validação.
    
    Args:
        input_file: Arquivo JSONL gerado pelo convert_for_fine_tuning.py
        train_ratio: Proporção dos dados para treinamento (0.8 = 80%)
        shuffle: Se deve embaralhar os dados antes da divisão
    """
    
    # Lê todos os dados
    print(f"📖 Lendo dados de {input_file}...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"Total de exemplos carregados: {len(data)}")
    
    # Embaralha se solicitado
    if shuffle:
        random.shuffle(data)
        print("✅ Dados embaralhados")
    
    # Calcula divisão
    total_examples = len(data)
    train_size = int(total_examples * train_ratio)
    val_size = total_examples - train_size
    
    # Divide os dados
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Gera nomes de arquivo baseados no arquivo de entrada
    base_name = input_file.replace('.jsonl', '')
    train_file = f"{base_name}_train.jsonl"
    val_file = f"{base_name}_validation.jsonl"
    
    # Salva dados de treinamento
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Salva dados de validação
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Imprime estatísticas
    print("\n📊 Divisão dos Dados:")
    print("=" * 40)
    print(f"Total de exemplos: {total_examples}")
    print(f"Treinamento: {train_size} exemplos ({train_ratio*100:.1f}%)")
    print(f"Validação: {val_size} exemplos ({(1-train_ratio)*100:.1f}%)")
    print()
    print(f"📁 Arquivos criados:")
    print(f"   Treinamento: {train_file}")
    print(f"   Validação: {val_file}")
    
    return train_file, val_file


def main():
    parser = argparse.ArgumentParser(description="Divide dados de fine-tuning em treinamento e validação")
    parser.add_argument('input_file', help='Arquivo JSONL gerado pelo convert_for_fine_tuning.py')
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='Proporção para treinamento (padrão: 0.8 = 80%%)')
    parser.add_argument('--no-shuffle', action='store_true',
                       help='Não embaralhar os dados antes da divisão')
    
    args = parser.parse_args()
    
    # Valida argumentos
    if not 0.1 <= args.train_ratio <= 0.9:
        print("❌ Erro: train_ratio deve estar entre 0.1 e 0.9")
        return
    
    try:
        train_file, val_file = split_fine_tuning_data(
            args.input_file, 
            args.train_ratio, 
            shuffle=not args.no_shuffle
        )
        
        print("\n✅ Divisão concluída com sucesso!")
        print("\n🚀 Próximos passos:")
        print(f"   openai api fine_tuning.jobs.create --training-file {train_file} --validation-file {val_file} --model gpt-3.5-turbo")
        
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo '{args.input_file}' não encontrado")
    except Exception as e:
        print(f"❌ Erro durante a divisão: {e}")


if __name__ == "__main__":
    main() 
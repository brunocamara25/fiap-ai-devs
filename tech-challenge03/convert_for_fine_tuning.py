#!/usr/bin/env python3
"""
Script para converter dados de produtos para formato de fine-tuning do ChatGPT.
Este script fornece múltiplas estratégias de conversão para diferentes objetivos de treinamento.
"""

import argparse
import json
import random
from typing import Any, Dict, List


def read_jsonl(file_path: str) -> List[Dict[str, str]]:
    """Lê arquivo JSONL e retorna lista de dicionários."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Erro analisando linha: {line[:100]}... Erro: {e}")
                    continue
    return data

def write_jsonl(data: List[Dict[str, Any]], output_path: str):
    """Escreve dados para arquivo JSONL."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def format_for_product_description_generation(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Formata dados para treinar um modelo para gerar descrições de produtos a partir de títulos.
    Caso de uso: Dado um título de produto, gerar uma descrição detalhada.
    """
    formatted_data = []
    
    for item in data:
        title = item.get('title', '').strip()
        content = item.get('content', '').strip()
        
        if not title or not content:
            continue
            
        # Cria um exemplo de treinamento
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional product description writer. Given a product title, write a compelling and detailed product description that highlights key features, benefits, and specifications."
                },
                {
                    "role": "user", 
                    "content": f"Write a product description for: {title}"
                },
                {
                    "role": "assistant",
                    "content": content
                }
            ]
        }
        formatted_data.append(training_example)
    
    return formatted_data

def format_for_product_qa(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Formata dados para treinar um modelo para responder perguntas sobre produtos.
    Caso de uso: Responder perguntas de clientes sobre características, especificações, etc.
    """
    formatted_data = []
    
    question_templates = [
        "What are the key features of this product?",
        "Can you tell me about this product?",
        "What makes this product special?",
        "Describe this product to me",
        "What should I know about this product?",
        "Give me details about this product",
        "What are the benefits of this product?",
        "Tell me about the specifications of this product"
    ]
    
    for item in data:
        title = item.get('title', '').strip()
        content = item.get('content', '').strip()
        
        if not title or not content:
            continue
        
        # Gera múltiplos pares P&R por produto
        question = random.choice(question_templates)
        
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful product specialist who answers customer questions about products based on their titles and descriptions. Provide accurate, helpful, and detailed responses."
                },
                {
                    "role": "user",
                    "content": f"Product: {title}\n\nQuestion: {question}"
                },
                {
                    "role": "assistant", 
                    "content": content
                }
            ]
        }
        formatted_data.append(training_example)
    
    return formatted_data

def format_for_product_improvement(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Formata dados para treinar um modelo para melhorar descrições de produtos.
    Caso de uso: Pegar informações básicas do produto e criar descrições aprimoradas.
    """
    formatted_data = []
    
    for item in data:
        title = item.get('title', '').strip()
        content = item.get('content', '').strip()
        
        if not title or not content:
            continue
        
        # Cria uma versão simplificada do título como entrada
        simplified_title = title.split(' - ')[0] if ' - ' in title else title
        
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert copywriter who improves product listings. Given a basic product title, create a comprehensive, engaging product description that highlights features, benefits, and appeals to customers."
                },
                {
                    "role": "user",
                    "content": f"Create an improved product description for: {simplified_title}"
                },
                {
                    "role": "assistant",
                    "content": f"**{title}**\n\n{content}"
                }
            ]
        }
        formatted_data.append(training_example)
    
    return formatted_data

def format_for_product_categorization(data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Formata dados para treinar um modelo para categorizar produtos.
    Caso de uso: Classificar produtos em categorias baseado em títulos e descrições.
    """
    formatted_data = []
    
    # Mapeamento simples de categorias baseado em palavras-chave
    category_keywords = {
        "Electronics": ["charger", "cable", "battery", "electronic", "usb", "bluetooth", "led", "screen", "phone", "iphone", "samsung", "sony", "camera", "tablet"],
        "Health & Beauty": ["cream", "serum", "beauty", "hair", "skin", "cosmetic", "makeup", "shampoo", "supplement", "vitamin", "oil", "lotion"],
        "Home & Garden": ["kitchen", "tool", "home", "garden", "storage", "holder", "organizer", "cleaner", "grill", "mat"],
        "Clothing & Accessories": ["shirt", "case", "cover", "backpack", "bag", "clothing", "apparel", "shoes"],
        "Sports & Outdoors": ["outdoor", "camping", "hiking", "sports", "exercise", "fitness", "bike", "hammock"],
        "Toys & Games": ["toy", "game", "dice", "figure", "children", "kids", "puzzle", "doll"],
        "Food & Beverages": ["food", "supplement", "vitamin", "nutrition", "protein", "organic"],
        "Other": []
    }
    
    for item in data:
        title = item.get('title', '').strip().lower()
        content = item.get('content', '').strip().lower()
        
        if not title or not content:
            continue
        
        # Determina categoria
        text_to_analyze = f"{title} {content}"
        category = "Other"
        
        for cat, keywords in category_keywords.items():
            if any(keyword in text_to_analyze for keyword in keywords):
                category = cat
                break
        
        training_example = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a product categorization specialist. Given a product title and description, classify it into one of these categories: Electronics, Health & Beauty, Home & Garden, Clothing & Accessories, Sports & Outdoors, Toys & Games, Food & Beverages, or Other."
                },
                {
                    "role": "user",
                    "content": f"Categorize this product:\nTitle: {item['title']}\nDescription: {item['content']}"
                },
                {
                    "role": "assistant",
                    "content": category
                }
            ]
        }
        formatted_data.append(training_example)
    
    return formatted_data

def main():
    parser = argparse.ArgumentParser(description='Converte dados de produto da Amazon para fine-tuning do ChatGPT')
    parser.add_argument('input_file', help='Caminho do arquivo JSONL de entrada')
    parser.add_argument('--output_dir', default='.', help='Diretório de saída (padrão: diretório atual)')
    parser.add_argument('--format', choices=['description', 'qa', 'improvement', 'categorization', 'all'], 
                       default='all', help='Formato de conversão (padrão: all)')
    parser.add_argument('--sample_size', type=int, help='Limitar número de amostras (para teste)')
    parser.add_argument('--shuffle', action='store_true', help='Embaralhar os dados antes do processamento')
    
    args = parser.parse_args()
    
    # Lê dados de entrada
    print(f"Lendo dados de {args.input_file}...")
    data = read_jsonl(args.input_file)
    print(f"Carregados {len(data)} itens")
    
    if args.shuffle:
        random.shuffle(data)
    
    if args.sample_size:
        data = data[:args.sample_size]
        print(f"Usando amostra de {len(data)} itens")
    
    # Conversões de formato
    formats_to_process = []
    if args.format == 'all':
        formats_to_process = ['description', 'qa', 'improvement', 'categorization']
    else:
        formats_to_process = [args.format]
    
    for format_type in formats_to_process:
        print(f"\nProcessando formato: {format_type}")
        
        if format_type == 'description':
            formatted_data = format_for_product_description_generation(data)
            output_file = f"{args.output_dir}/fine_tune_product_description.jsonl"
            
        elif format_type == 'qa':
            formatted_data = format_for_product_qa(data)
            output_file = f"{args.output_dir}/fine_tune_product_qa.jsonl"
            
        elif format_type == 'improvement':
            formatted_data = format_for_product_improvement(data)
            output_file = f"{args.output_dir}/fine_tune_product_improvement.jsonl"
            
        elif format_type == 'categorization':
            formatted_data = format_for_product_categorization(data)
            output_file = f"{args.output_dir}/fine_tune_product_categorization.jsonl"
            
        print(f"Gerados {len(formatted_data)} exemplos de treinamento")
        write_jsonl(formatted_data, output_file)
        print(f"Salvo em: {output_file}")
    
    print("\n✅ Conversão concluída!")
    print("\nPróximos passos:")
    print("1. Revisar os arquivos gerados")
    print("2. Dividir dados com: python split_data.py <arquivo>")
    print("3. Fazer upload para OpenAI usando: openai api fine_tuning.jobs.create --training-file <arquivo>")
    print("4. Monitorar progresso do treinamento")
    print("5. Testar seu modelo fine-tuned")

if __name__ == "__main__":
    main() 
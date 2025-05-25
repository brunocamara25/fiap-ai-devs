#!/usr/bin/env python3
"""
Script de Limpeza de Dados
Limpa e filtra dados de produtos da Amazon para melhores resultados de fine-tuning.
"""

import argparse
import html
import json
import re
import unicodedata
from collections import defaultdict
from typing import Dict, List, Set


class AmazonDataCleaner:
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'html_entities_cleaned': 0,
            'duplicates_removed': 0,
            'short_content_removed': 0,
            'long_title_truncated': 0,
            'marketing_spam_removed': 0,
            'final_count': 0
        }
        
        # Padrões para identificar spam de marketing
        self.spam_patterns = [
            r'^\d+% OFF',
            r'#\d+\s+BEST',
            r'★.*★',
            r'⭐.*⭐',
            r'🎯|🚀|✅|❌|💡|⚡',
            r'MONEY BACK GUARANTEE',
            r'LIMITED TIME',
            r'CLICK.*ADD TO CART',
            r'BUY NOW',
        ]
        
    def clean_html_entities(self, text: str) -> str:
        """Limpa entidades HTML e normaliza o texto."""
        if not text:
            return ""
            
        # Decodifica entidades HTML
        text = html.unescape(text)
        
        # Substituições específicas adicionais
        entity_map = {
            '&reg;': '®',
            '&amp;': '&',
            '&quot;': '"',
            '&#x2022;': '•',
            '&#9733;': '★',
            '&nbsp;': ' ',
            '&lt;': '<',
            '&gt;': '>',
        }
        
        for entity, replacement in entity_map.items():
            text = text.replace(entity, replacement)
            
        # Remove símbolos excessivos e normaliza
        text = re.sub(r'[★⭐]{2,}', '★', text)  # Múltiplas estrelas para única
        text = re.sub(r'[•]{2,}', '•', text)   # Múltiplos bullets para único
        text = re.sub(r'\s+', ' ', text)       # Múltiplos espaços para único
        text = text.strip()
        
        return text
    
    def is_marketing_spam(self, title: str) -> bool:
        """Detecta se o título contém spam de marketing excessivo."""
        title_upper = title.upper()
        
        spam_indicators = 0
        for pattern in self.spam_patterns:
            if re.search(pattern, title_upper):
                spam_indicators += 1
                
        # Verifica maiúsculas excessivas (>50% das letras são maiúsculas)
        letters = [c for c in title if c.isalpha()]
        if letters:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > 0.5:
                spam_indicators += 1
                
        # Verifica comprimento do título (marketing excessivo tende a ser muito longo)
        if len(title) > 200:
            spam_indicators += 1
            
        return spam_indicators >= 2
    
    def clean_title(self, title: str) -> str:
        """Limpa e potencialmente trunca o título do produto."""
        title = self.clean_html_entities(title)
        
        # Remove lixo de marketing do início/final
        title = re.sub(r'^[\d%\s]*OFF\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^\s*#\d+\s*BEST\s*', '', title, flags=re.IGNORECASE)
        
        # Trunca títulos muito longos em pontos naturais de quebra
        if len(title) > 150:
            # Tenta quebrar em pontos naturais
            break_points = [' - ', ' | ', ' + ', ' with ', ' for ']
            for bp in break_points:
                if bp in title and title.index(bp) < 150:
                    title = title[:title.index(bp)]
                    self.stats['long_title_truncated'] += 1
                    break
            else:
                # Nenhuma quebra natural encontrada, trunca na fronteira da palavra
                if len(title) > 150:
                    title = title[:147] + "..."
                    self.stats['long_title_truncated'] += 1
                    
        return title.strip()
    
    def clean_content(self, content: str) -> str:
        """Limpa conteúdo/descrição do produto."""
        content = self.clean_html_entities(content)
        
        # Remove frases comuns de marketing
        marketing_remove = [
            r'Click.*add to cart.*',
            r'Buy now.*',
            r'Order today.*',
            r'Limited time.*',
            r'Don\'t wait.*',
            r'Act now.*',
        ]
        
        for pattern in marketing_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
        # Limpa pontuação excessiva
        content = re.sub(r'[!]{2,}', '!', content)
        content = re.sub(r'[?]{2,}', '?', content)
        content = re.sub(r'\.{3,}', '...', content)
        
        # Normaliza espaços em branco
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        return content
    
    def is_valid_entry(self, title: str, content: str) -> bool:
        """Verifica se a entrada atende aos padrões de qualidade."""
        # Pula entradas com conteúdo ausente ou muito curto
        if not content or len(content.strip()) < 20:
            return False
            
        # Pula entradas com títulos ausentes ou muito curtos  
        if not title or len(title.strip()) < 10:
            return False
            
        # Pula spam de marketing
        if self.is_marketing_spam(title):
            return False
            
        # Pula se o conteúdo é apenas placeholder
        if content.strip() in ['...', '..', '.', 'Description:', '']:
            return False
            
        return True
    
    def generate_content_hash(self, title: str, content: str) -> str:
        """Gera hash para detecção de duplicatas."""
        # Normaliza para comparação
        normalized = (title + content).lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    def clean_data(self, input_file: str, output_file: str, 
                   remove_duplicates: bool = True,
                   min_content_length: int = 20,
                   max_title_length: int = 150) -> Dict:
        """Função principal de limpeza."""
        
        print(f"🧹 Iniciando limpeza de dados para {input_file}")
        
        seen_hashes: Set[str] = set()
        cleaned_data = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    self.stats['total_processed'] += 1
                    
                    # Extrai e limpa campos
                    title = data.get('title', '').strip()
                    content = data.get('content', '').strip()
                    
                    # Limpa o texto
                    original_title = title
                    original_content = content
                    
                    title = self.clean_title(title)
                    content = self.clean_content(content)
                    
                    # Rastreia estatísticas de limpeza
                    if original_title != title or original_content != content:
                        self.stats['html_entities_cleaned'] += 1
                    
                    # Valida qualidade da entrada
                    if not self.is_valid_entry(title, content):
                        if len(content) < min_content_length:
                            self.stats['short_content_removed'] += 1
                        elif self.is_marketing_spam(title):
                            self.stats['marketing_spam_removed'] += 1
                        continue
                    
                    # Verifica duplicatas
                    if remove_duplicates:
                        content_hash = self.generate_content_hash(title, content)
                        if content_hash in seen_hashes:
                            self.stats['duplicates_removed'] += 1
                            continue
                        seen_hashes.add(content_hash)
                    
                    # Adiciona entrada limpa
                    cleaned_entry = {
                        'title': title,
                        'content': content
                    }
                    cleaned_data.append(cleaned_entry)
                    
                    if line_num % 1000 == 0:
                        print(f"Processadas {line_num} linhas, mantidas {len(cleaned_data)} entradas")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠️  Erro JSON na linha {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"⚠️  Erro processando linha {line_num}: {e}")
                    continue
        
        # Escreve dados limpos
        self.stats['final_count'] = len(cleaned_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in cleaned_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        return self.stats
    
    def print_stats(self):
        """Imprime estatísticas de limpeza."""
        print("\n📊 Resultados da Limpeza de Dados:")
        print("=" * 50)
        print(f"Total de entradas processadas: {self.stats['total_processed']:,}")
        print(f"Entidades HTML limpas: {self.stats['html_entities_cleaned']:,}")
        print(f"Duplicatas removidas: {self.stats['duplicates_removed']:,}")
        print(f"Conteúdo curto removido: {self.stats['short_content_removed']:,}")
        print(f"Títulos longos truncados: {self.stats['long_title_truncated']:,}")
        print(f"Spam de marketing removido: {self.stats['marketing_spam_removed']:,}")
        print(f"Entradas limpas finais: {self.stats['final_count']:,}")
        
        if self.stats['total_processed'] > 0:
            retention_rate = (self.stats['final_count'] / self.stats['total_processed']) * 100
            print(f"Taxa de retenção de dados: {retention_rate:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="Limpa dados de produtos Amazon para fine-tuning")
    parser.add_argument('input_file', help='Arquivo JSONL de entrada')
    parser.add_argument('-o', '--output', default='cleaned_amazon_data.jsonl', 
                       help='Nome do arquivo de saída (padrão: cleaned_amazon_data.jsonl)')
    parser.add_argument('--keep-duplicates', action='store_true',
                       help='Manter entradas duplicadas (padrão: remover duplicatas)')
    parser.add_argument('--min-content-length', type=int, default=20,
                       help='Comprimento mínimo do conteúdo (padrão: 20)')
    parser.add_argument('--max-title-length', type=int, default=150,
                       help='Comprimento máximo do título antes do truncamento (padrão: 150)')
    
    args = parser.parse_args()
    
    cleaner = AmazonDataCleaner()
    
    try:
        cleaner.clean_data(
            args.input_file, 
            args.output,
            remove_duplicates=not args.keep_duplicates,
            min_content_length=args.min_content_length,
            max_title_length=args.max_title_length
        )
        cleaner.print_stats()
        print(f"\n✅ Dados limpos salvos em: {args.output}")
        
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo '{args.input_file}' não encontrado")
    except Exception as e:
        print(f"❌ Erro durante a limpeza: {e}")

if __name__ == "__main__":
    main() 
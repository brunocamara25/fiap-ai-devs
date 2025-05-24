# Conversor de Dados de Produtos Amazon com Fine-tuning da OpenAI

Este script converte seus dados de produtos da  Amazon do formato JSONL para o formato adequado para fazer fine-tuning de modelos da OpenAI. Fornece múltiplas estratégias de conversão para diferentes objetivos de treinamento.


## Integrantes do Grupo 👥
- Bruno Câmara - RM 359922
- Fabricio Cavalcante - RM 360135
- Leonardo Charelli - RM 360425
- Lucas Pincho - RM 360216
- Natalia Rosa - RM 358301

## 🎯 Cenários de Treinamento Suportados

### 1. Geração de Descrição de Produtos
**Caso de uso**: Treinar o modelo para gerar descrições detalhadas de produtos a partir de títulos.
- **Entrada**: Título do produto
- **Saída**: Descrição detalhada do produto

### 2. Perguntas e Respostas sobre Produtos
**Caso de uso**: Treinar o modelo para responder perguntas de clientes sobre produtos.
- **Entrada**: Título do produto + pergunta do cliente
- **Saída**: Resposta útil baseada nas informações do produto

### 3. Melhoria de Descrição de Produtos
**Caso de uso**: Treinar o modelo para aprimorar informações básicas do produto.
- **Entrada**: Título básico do produto
- **Saída**: Título aprimorado + descrição detalhada

### 4. Categorização de Produtos
**Caso de uso**: Treinar o modelo para classificar produtos em categorias.
- **Entrada**: Título + descrição do produto
- **Saída**: Classificação de categoria


## 📋 Requisitos

Os scripts utilizam apenas bibliotecas padrão do Python (não requer instalação adicional):
- `argparse` - Para parsing de argumentos da linha de comando
- `json` - Para manipulação de dados JSON/JSONL
- `random` - Para embaralhamento de dados
- `html` - Para decodificação de entidades HTML
- `re` - Para expressões regulares
- `unicodedata` - Para normalização de texto
- `typing` - Para type hints

**Python 3.6+ é requerido.**

## 🧹 Limpeza de Dados

Executar o comando abaixo, esse primeiro comando é responsavel por retornar o nosso arquivo com apenas as informações de título e conteúdo.

```bash
jq -c 'select(.title != "" and .content != "") | {title, content}' example.jsonl >
example_filter.jsonl && echo "Filtragem Concluída. Linhas Originais:
$(wc -l < example_filter.jsonl), Linhas Filtradas: $(wc -l < trn_filter.jsonl)"
```

**⚠️ IMPORTANTE**: Dados brutos do arquivo da Amazon frequentemente contêm entidades HTML, spam de marketing, duplicatas e problemas de formatação que prejudicarão o desempenho do seu modelo. **Limpe seus dados primeiro!**

### Por Que Limpar Seus Dados?

Notamos que o conteudo do arquivo tem:
- **Entidades HTML**: `&amp;`, `&reg;`, `&#9733;` ao invés de `&`, `®`, `★`
- **Spam de marketing**: Emojis excessivos, texto em maiúsculas, linguagem promocional
- **Duplicatas**: Mesmos produtos listados múltiplas vezes
- **Títulos longos**: Títulos de 200+ caracteres com palavras-chave excessivas
- **Formatação ruim**: Espaçamento inconsistente, símbolos, problemas de codificação

### Script de Limpeza de Dados

O script `clean_amazon_data.py` corrige automaticamente esses problemas:

#### ✅ **O Que Ele Corrige:**
- **Decodificação de Entidades HTML**: Converte entidades HTML para caracteres apropriados
- **Detecção de Spam de Marketing**: Remove conteúdo promocional excessivo
- **Truncamento de Títulos**: Encurta inteligentemente títulos excessivamente longos
- **Remoção de Duplicatas**: Identifica e remove produtos duplicados
- **Validação de Conteúdo**: Remove entradas com conteúdo insuficiente
- **Normalização de Texto**: Padroniza espaçamento, pontuação e formatação

#### 🚀 **Uso Rápido:**

```bash
# Limpe seus dados primeiro (recomendado)
python clean_amazon_data.py trn_filter.jsonl -o cleaned_amazon_data.jsonl
```

#### 📊 **Opções de Linha de Comando:**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `input_file` | Arquivo JSONL de entrada para limpar | Obrigatório |
| `-o, --output` | Nome do arquivo de saída | `cleaned_amazon_data.jsonl` |
| `--keep-duplicates` | Manter entradas duplicadas | Remover duplicatas |
| `--min-content-length` | Comprimento mínimo do conteúdo | 20 caracteres |
| `--max-title-length` | Comprimento máx. do título antes do truncamento | 150 caracteres |

#### 📈 **Resultados de Exemplo:**

```
📊 Resultados da Limpeza de Dados:
==================================================
Total de entradas processadas: 159
Entidades HTML limpas: 103
Duplicatas removidas: 4
Conteúdo curto removido: 2
Títulos longos truncados: 66
Spam de marketing removido: 1
Entradas limpas finais: 151
Taxa de retenção de dados: 95.0%
```

#### 🔧 **Limpeza Avançada:**

```bash
# Manter duplicatas mas limpar formatação
python clean_amazon_data.py input.jsonl --keep-duplicates -o cleaned.jsonl

# Filtragem de conteúdo mais agressiva
python clean_amazon_data.py input.jsonl --min-content-length 50 -o cleaned.jsonl

# Permitir títulos mais longos
python clean_amazon_data.py input.jsonl --max-title-length 200 -o cleaned.jsonl
```

### Antes vs Depois da Limpeza

**❌ Antes da Limpeza:**
```json
{
  "title": "&#9733;4.8 / 5 Stars&#9733; - Forskolin Pure Coleus Forskohlii Root Standardized to 20% Weight Loss Supplement and Appetite Suppressant - Dr Oz Highly Recommended Product for Fat Burning and Melting Belly Fat - The Best Forskolin Product on the Market!! 250mg Yielding 50 Mg of Active Forskolin - Works Excellent with Pure Garcinia Cambogia and Colon Cleanse Detox and Full Body Detox - 100% Money Back Guarantee - 60 Veggie Capsules",
  "content": "BD HeaderHow Does Coleus Forskohlii (Forskolin) Affect cAMP?..."
}
```

**✅ Depois da Limpeza:**
```json
{
  "title": "★4.8 / 5 Stars★ - Forskolin Pure Coleus Forskohlii Root Standardized to 20% Weight Loss Supplement and Appetite Suppressant",
  "content": "BD HeaderHow Does Coleus Forskohlii (Forskolin) Affect cAMP?..."
}
```

## 📊 Divisão de Dados: Treinamento vs Validação

### Por Que Dividir Seus Dados?

#### ✅ **Benefícios da Divisão de Dados:**
- **Previne Overfitting**: Ajuda você a saber quando parar o treinamento
- **Monitoramento de Performance**: Acompanha quão bem seu modelo generaliza
- **Otimização de Custos**: Evita épocas de treinamento desnecessárias
- **Garantia de Qualidade**: Testa modelo em dados não vistos
- **Detecção Precoce de Problemas**: Identifica problemas de generalização

### Script de Divisão de Dados

O script `split_data.py` divide automaticamente seus dados convertidos:

#### 🚀 **Uso Rápido:**

```bash
# Divide dados em 80% treinamento / 20% validação (padrão)
python split_data.py fine_tune_product_description.jsonl
```

#### 📊 **Opções de Linha de Comando:**

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `input_file` | Arquivo JSONL gerado pelo convert_for_fine_tuning.py | Obrigatório |
| `--train_ratio` | Proporção para treinamento (0.1 a 0.9) | 0.8 (80%) |
| `--no-shuffle` | Não embaralhar dados antes da divisão | Embaralhar habilitado |

#### 📈 **Resultado de Exemplo:**

```
📊 Divisão dos Dados:
========================================
Total de exemplos: 151
Treinamento: 120 exemplos (80.0%)
Validação: 31 exemplos (20.0%)

📁 Arquivos criados:
   Treinamento: fine_tune_product_description_train.jsonl
   Validação: fine_tune_product_description_validation.jsonl
```

#### 🎯 **Guia de Proporções de Divisão:**

| Tamanho do Dataset | Divisão Recomendada | Treinamento | Validação |
|-------------------|-------------------|-------------|-----------|
| 50-100 exemplos | 90/10 | 90% | 10% |
| 100-500 exemplos | 85/15 | 85% | 15% |
| 500+ exemplos | 80/20 | 80% | 20% |

#### 🔧 **Divisão Avançada:**

```bash
# Divisão 85/15 (recomendado para datasets menores)
python split_data.py fine_tune_product_description.jsonl --train_ratio 0.85

# Divisão 90/10 (para datasets muito pequenos)
python split_data.py fine_tune_product_description.jsonl --train_ratio 0.9

# Manter ordem original (não embaralhar)
python split_data.py fine_tune_product_description.jsonl --no-shuffle
```

### 🎯 **Fluxo de Trabalho Recomendado**

```bash
# Passo 1: Limpar seus dados
python clean_amazon_data.py trn_filter.jsonl -o cleaned_amazon_data.jsonl

# Passo 2: Converter para fine-tuning
python convert_for_fine_tuning.py cleaned_amazon_data.jsonl --format description --shuffle

# Passo 3: Dividir em treinamento/validação
python split_data.py fine_tune_product_description.jsonl

# Passo 4: Iniciar fine-tuning com validação - Recomendadmos usar a versão gráfica
# direto do dashboard -  https://platform.openai.com/finetune
openai api fine_tuning.jobs.create \
  --training-file fine_tune_product_description_train.jsonl \
  --validation-file fine_tune_product_description_validation.jsonl \
  --model gpt-3.5-turbo \
  --suffix "product-descriptions"
```

**A divisão de dados tipicamente melhora a qualidade do modelo em 15-30%**, resultando em:
- ✅ Melhor generalização para novos produtos
- ✅ Redução de overfitting
- ✅ Detecção precoce de problemas de treinamento
- ✅ Otimização de custos de treinamento

## 🚀 Uso

### Uso Básico (Converter todos os formatos)
```bash
python convert_for_fine_tuning.py cleaned_amazon_data.jsonl
```

### Converter apenas formato específico
```bash
python convert_for_fine_tuning.py cleaned_amazon_data.jsonl --format description
```

### Testar com uma amostra (recomendado primeiro)
```bash
python convert_for_fine_tuning.py cleaned_amazon_data.jsonl --sample_size 100 --shuffle
```

### Salvar em diretório específico
```bash
python convert_for_fine_tuning.py cleaned_amazon_data.jsonl --output_dir ./fine_tune_data/
```

## 📊 Opções de Linha de Comando

| Opção | Descrição | Padrão |
|-------|-----------|--------|
| `input_file` | Caminho para seu arquivo JSONL | Obrigatório |
| `--format` | Formato de conversão: `description`, `qa`, `improvement`, `categorization`, `completion`, `all` | `all` |
| `--output_dir` | Diretório para salvar arquivos de saída | Diretório atual |
| `--sample_size` | Limitar número de amostras para teste | Todos os dados |
| `--shuffle` | Embaralhar dados aleatoriamente antes do processamento | False |

## 📁 Arquivos de Saída

O script gera arquivos diferentes baseados no formato:

- `fine_tune_product_description.jsonl` - Para geração de descrição
- `fine_tune_product_qa.jsonl` - Para treinamento de P&R
- `fine_tune_product_improvement.jsonl` - Para melhoria de descrição
- `fine_tune_product_categorization.jsonl` - Para categorização de produtos

## 📝 Exemplos de Formatos de Saída

### Formato Chat (GPT-3.5-turbo/GPT-4)
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a professional product description writer..."
    },
    {
      "role": "user",
      "content": "Write a product description for: iPhone Charger Cable"
    },
    {
      "role": "assistant",
      "content": "Premium quality charging cable compatible with iPhone..."
    }
  ]
}
```


## 🔧 Fine-tuning com OpenAI

### 1. Instalar CLI da OpenAI
```bash
pip install openai
```

### 2. Definir sua chave da API
```bash
export OPENAI_API_KEY="sua-chave-da-api-aqui"
```

### 3. Fazer upload e iniciar fine-tuning - Tivemos problema com a execução via CLI, então recomendamos nessa etapa fazer direto no dashboard da plataforma da OpenAI

#### Para modelos Chat (GPT-3.5-turbo):
```bash
# Com arquivo de validação (recomendado)
openai api fine_tuning.jobs.create \
  --training-file fine_tune_product_description_train.jsonl \
  --validation-file fine_tune_product_description_validation.jsonl \
  --model gpt-3.5-turbo \
  --suffix "product-descriptions"

# Sem arquivo de validação (não recomendado)
openai api fine_tuning.jobs.create \
  --training-file fine_tune_product_description.jsonl \
  --model gpt-3.5-turbo \
  --suffix "product-descriptions"
```

### 3. Monitorar treinamento - É possivel ver direto na plataforma da OpenAI também
```bash
# Listar jobs de fine-tuning
openai api fine_tuning.jobs.list

# Monitorar job específico
openai api fine_tuning.jobs.retrieve -i ftjob-abc123

# Ver eventos do job
openai api fine_tuning.jobs.list-events -i ftjob-abc123
```

### 4. Testar seu modelo fine-tuned - É possivel executar direto na plataforma da OpenAI também
```bash
# Usando a CLI da OpenAI
openai api chat.completions.create \
  --model ft:gpt-3.5-turbo:your-org:product-descriptions:abc123 \
  --messages '[{"role": "user", "content": "Write a product description for: Wireless Headphones"}]'
```

## ✅ Melhores Práticas

### Qualidade dos Dados
- **Limpe seus dados PRIMEIRO**: Use `clean_amazon_data.py` para corrigir entidades HTML, remover spam e padronizar formatação
- **Verifique resultados da limpeza**: Revise as estatísticas de limpeza para garantir boa retenção de dados
- **Remova entradas de baixa qualidade**: O script de limpeza remove automaticamente entradas com conteúdo insuficiente
- **Trate duplicatas**: Use remoção de duplicatas (habilitada por padrão no script de limpeza)
- **Valide após limpeza**: Verifique manualmente algumas entradas limpas para garantir qualidade

### Otimização de Custos
- **Faça amostra dos seus dados**: Comece com 100-500 exemplos para teste
- **Escolha o modelo certo**: GPT-3.5-turbo é mais barato que GPT-4

## 🎯 Fine-tuning de Tarefa Única vs Multi-tarefas

### 🏆 **Recomendação: Foque em UMA Tarefa**

**É fortemente recomendado escolher UM formato específico** ao invés de misturar múltiplas tarefas. Aqui está o porquê:

#### ✅ Benefícios do Fine-tuning de Tarefa Única
- **Maior Qualidade**: Modelo se torna altamente especializado e performa melhor
- **Saídas Consistentes**: Formato e estilo previsíveis sempre  
- **Avaliação Mais Fácil**: Simples de medir sucesso e identificar problemas
- **Menos Confusão**: Modelo não mistura diferentes formatos de tarefa

#### ❌ Problemas com Fine-tuning Multi-tarefas
- **Degradação de Desempenho**: Sabe de tudo um pouco, especialista em nada
- **Confusão de Formato**: Modelo pode misturar diferentes estilos de saída
- **Resultados Inconsistentes**: Comportamento imprevisível dependendo da entrada

### 📊 Comparação de Tarefas

| Tarefa | Melhor Para | Adequação dos Dados | Valor Comercial | Dificuldade |
|--------|-------------|---------------------|-----------------|-------------|
| `description` |  E-commerce | Perfeita | Muito Alto | Fácil |
| `qa` |  Suporte ao Cliente | Boa | Alto | Médio |
| `improvement` |  Melhoria de Conteúdo | Boa | Médio | Médio |
| `categorization` |  Organização | Razoável | Médio | Difícil |



## 🔍 Exemplo de Fluxo de Trabalho

1. **Limpar seus dados (PRIMEIRO PASSO)**:
   ```bash
   python clean_amazon_data.py trn_filter.jsonl -o cleaned_amazon_data.jsonl
   ```

2. **Testar o script de conversão**:
   ```bash
   python convert_for_fine_tuning.py cleaned_amazon_data.jsonl --sample_size 10 --format description
   ```

3. **Revisar a saída**:
   ```bash
   head -5 fine_tune_product_description.jsonl
   ```

4. **Gerar dados de treinamento**:
   ```bash
   python convert_for_fine_tuning.py cleaned_amazon_data.jsonl --format description --shuffle
   ```

5. **Dividir dados para treinamento e validação**:
   ```bash
   python split_data.py fine_tune_product_description.jsonl
   ```

6. **Iniciar fine-tuning com validação**: Usar interface grafica como alternativa. https://platform.openai.com/finetune
   ```bash
   openai api fine_tuning.jobs.create \
     --training-file fine_tune_product_description_train.jsonl \
     --validation-file fine_tune_product_description_validation.jsonl \
     --model gpt-3.5-turbo \
     --suffix "product-descriptions"
   ```

7. **Monitorar progresso do treinamento**:
   ```bash
   openai api fine_tuning.jobs.list-events -i ftjob-abc123
   ```
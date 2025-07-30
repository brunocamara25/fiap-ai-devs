# 🎥 Sistema de Análise de Vídeo com IA — Tech Challenge FIAP

## Sumário

- [Descrição](#descrição)
- [Funcionalidades](#funcionalidades)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Usar](#como-usar)
- [Argumentos de Linha de Comando](#argumentos-de-linha-de-comando)
- [Fluxo de Execução](#fluxo-de-execução)
- [Detalhes de Implementação](#detalhes-de-implementação)
- [Exemplo de Execução](#exemplo-de-execução)
- [Saída e Relatórios](#saída-e-relatórios)
- [Requisitos](#requisitos)
- [Licença](#licença)

---

## Descrição

Este projeto é um sistema completo para análise de vídeos utilizando Inteligência Artificial, desenvolvido como parte do Tech Challenge da FIAP. O sistema processa vídeos, detecta rostos, estima pessoas, identifica anomalias e gera relatórios detalhados sobre o conteúdo analisado.

---

## Funcionalidades

- **Validação automática** do arquivo de vídeo (extensão, existência, tamanho)
- **Processamento inteligente** de frames (intervalo configurável)
- **Detecção de rostos e pessoas**
- **Identificação de anomalias**
- **Geração de relatórios** em múltiplos formatos
- **Logs detalhados** e exportação de resultados
- **Opção de salvar vídeo anotado** com detecções

---

## Estrutura do Projeto

```
├── main.py # Script principal
├── src/
│ ├── video_analyzer.py # Módulo de análise de vídeo
│ └── report_generator.py # Módulo de geração de relatórios
├── requirements.txt # Dependências do projeto
├── data/
│ └── output/ # Saída dos resultados
├── logs/ # Logs de execução
└── README.md # Este arquivo
```
## Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/seu-repo.git
   cd seu-repo
   ```

2. **Crie e ative um ambiente virtual:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Linux/Mac
   venv\Scripts\activate         # Windows
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## Como Usar

### Execução básica
```bash
python main.py caminho/para/video.mp4
```

### Opções avançadas

- **Definir diretório de saída:**
  ```bash
  python main.py video.mp4 --output-dir ./resultados
  ```

- **Alterar intervalo de processamento:**
  ```bash
  python main.py video.mp4 --interval 10
  ```

- **Salvar vídeo anotado:**
  ```bash
  python main.py video.mp4 --save-video
  ```

- **Ajustar nível de log:**
  ```bash
  python main.py video.mp4 --log-level DEBUG
  ```

- **Pular geração de relatórios:**
  ```bash
  python main.py video.mp4 --no-reports
  ```

## Argumentos de Linha de Comando
| Argumento      | Descrição                                         | Padrão        |
|----------------|---------------------------------------------------|---------------|
| `video_path`   | Caminho para o vídeo a ser analisado              | **Obrigatório**|
| `--output-dir` | Diretório base para salvar resultados             | `data/output` |
| `--interval`   | Processar a cada N frames                         | `5`           |
| `--save-video` | Salvar vídeo anotado                              | `False`       |
| `--log-level`  | Nível de log (`DEBUG`, `INFO`, `WARNING`, `ERROR`)| `INFO`        |
| `--no-reports` | Pular geração de relatórios                       | `False`       |

## Fluxo de Execução

1. **Validação do vídeo:** O sistema verifica se o arquivo existe, se a extensão é suportada e se não está vazio.
2. **Criação da estrutura de saída:** Diretórios para relatórios, visualizações e dados são criados automaticamente.
3. **Processamento do vídeo:** O vídeo é analisado frame a frame (com intervalo configurável), detectando rostos, pessoas e anomalias.
4. **Geração de relatórios:** Relatórios são gerados em múltiplos formatos e salvos na pasta de saída.
5. **Resumo e logs:** Um resumo da análise é salvo em JSON e logs detalhados são mantidos na pasta `logs/`.## Saída e Relatórios

Após a execução, os resultados serão salvos em um diretório como `data/output/analysis_YYYYMMDD_HHMMSS/`, contendo:

- `reports/` — Relatórios gerados (ex: CSV, JSON, PDF)
- `visualizations/` — Visualizações e gráficos
- `data/` — Dados brutos da análise
- `analysis_summary.json` — Resumo da análise
- `error_log.json` — (se houver erro)

## Detalhes de Implementação

### 1. **Validação do Vídeo**
- **Função:** `validate_video_file`
- **Descrição:** Verifica existência, extensão e tamanho do arquivo. Gera erro se o arquivo não for válido.

### 2. **Estrutura de Saída**
- **Função:** `create_output_structure`
- **Descrição:** Cria diretórios para relatórios, visualizações e dados, organizando os resultados por data/hora.

### 3. **Processamento e Análise**
- **Classe:** `VideoAnalyzer` (em `src/video_analyzer.py`)
- **Descrição:** Processa o vídeo, detecta rostos, pessoas e anomalias. Permite salvar vídeo anotado.

### 4. **Geração de Relatórios**
- **Classe:** `ReportGenerator` (em `src/report_generator.py`)
- **Descrição:** Gera relatórios a partir dos resultados da análise, em diferentes formatos.

### 5. **Logging**
- **Função:** `setup_logging`
- **Descrição:** Logs são salvos em `logs/` e exibidos no terminal, facilitando o rastreio de erros e execuções.

### 6. **Resumo dos Resultados**
- **Função:** `print_results_summary`
- **Descrição:** Exibe no terminal um resumo dos principais achados e arquivos gerados.

## Exemplo de Execução

```bash
python main.py video.mp4 --interval 10 --save-video --output-dir ./resultados --log-level DEBUG
```

## Saída e Relatórios:
```
🔍 Validando arquivo de vídeo...
✅ Arquivo válido: video.mp4
📁 Diretório de saída: data/output/analysis_20240610_153000
🚀 Iniciando análise...
• Intervalo de processamento: 1 a cada 10 frames
• Salvar vídeo anotado: Sim
⏳ Isso pode demorar alguns minutos...

📊 RESUMO DA ANÁLISE
✅ Status: SUCESSO
⏱️ Tempo de processamento: 2.3 minutos
📁 Resultados salvos em: data/output/analysis_20240610_153000

📈 ESTATÍSTICAS:
• Frames totais: 3000
• Frames analisados: 300
• Rostos detectados: 120
• Pessoas estimadas: 15
• Anomalias encontradas: 2

🎯 DESCOBERTAS PRINCIPAIS:
• Pessoa desconhecida detectada em 00:01:23
• Comportamento anômalo em 00:02:10

📄 ARQUIVOS GERADOS: • Dados brutos: results.json • Relatório CSV: report.csv • Relatório PDF: report.pdf
```

## Requisitos

- Python 3.10+
- Dependências listadas em `requirements.txt`

## Licença

Projeto desenvolvido para fins educacionais no Tech Challenge - FIAP.

> **Dica:** Para dúvidas sobre uso, execute:
> ```bash
> python main.py --help
> ```
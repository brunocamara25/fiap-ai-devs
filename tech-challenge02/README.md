# Otimizador de Portfólio com Algoritmo Genético

Uma aplicação web em Streamlit que utiliza algoritmos genéticos para otimizar a alocação de portfólio de investimentos baseado no Índice Sharpe. Esta ferramenta ajuda investidores a encontrar pesos ótimos para o portfólio através da análise de dados históricos de ações e maximização dos retornos ajustados ao risco.

## Funcionalidades

- **
Otimização de Portfólio
**:
  - Algoritmos genéticos para encontrar a melhor alocação de ativos.
  - Suporte a multiobjetivo (retorno e risco).
- **
Configuração Personalizável
**:
  - Valor do investimento, período de análise e seleção de ações.
  - Ajuste de parâmetros do algoritmo (população, gerações, taxa de mutação, etc.).
- **
Visualizações Interativas
**:
  - Gráficos de progresso da otimização.
  - Alocação do portfólio em gráficos de pizza.
  - Evolução do Pareto Front para análises multiobjetivo.
- **
Métricas Detalhadas
**:
  - Retorno esperado, volatilidade e Índice Sharpe.
  - Projeções de investimento para diferentes horizontes temporais.
- **
Exportação de Resultados
**:
  - Baixe os resultados do portfólio em formato CSV.

## Estrutura do Projeto

O projeto foi modularizado para facilitar a manutenção e escalabilidade:

- `app.py`: Arquivo principal que gerencia a interface do usuário com Streamlit.
- `genetic_algorithm.py`: Implementação do algoritmo genético para otimização do portfólio.
- `data.py`: Funções para download e tratamento de dados históricos de ações.
- `metrics.py`: Cálculo de métricas financeiras como Índice Sharpe, Sortino, Treynor, etc.
- `visualization.py`: Funções para exibição de gráficos e tabelas interativas.

## Instalação# Otimizador de Portfólio com Algoritmo Genético

Uma aplicação web em Streamlit que utiliza algoritmos genéticos para otimizar a alocação de portfólio de investimentos baseado no Índice Sharpe. Esta ferramenta ajuda investidores a encontrar pesos ótimos para o portfólio através da análise de dados históricos de ações e maximização dos retornos ajustados ao risco.

## Funcionalidades

- **
Otimização de Portfólio
**:
  - Algoritmos genéticos para encontrar a melhor alocação de ativos.
  - Suporte a multiobjetivo (retorno e risco).
- **
Configuração Personalizável
**:
  - Valor do investimento, período de análise e seleção de ações.
  - Ajuste de parâmetros do algoritmo (população, gerações, taxa de mutação, etc.).
- **
Visualizações Interativas
**:
  - Gráficos de progresso da otimização.
  - Alocação do portfólio em gráficos de pizza.
  - Evolução do Pareto Front para análises multiobjetivo.
- **
Métricas Detalhadas
**:
  - Retorno esperado, volatilidade e Índice Sharpe.
  - Projeções de investimento para diferentes horizontes temporais.
- **
Exportação de Resultados
**:
  - Baixe os resultados do portfólio em formato CSV.

## Estrutura do Projeto

O projeto foi modularizado para facilitar a manutenção e escalabilidade:

- `app.py`: Arquivo principal que gerencia a interface do usuário com Streamlit.
- `genetic_algorithm.py`: Implementação do algoritmo genético para otimização do portfólio.
- `data.py`: Funções para download e tratamento de dados históricos de ações.
- `metrics.py`: Cálculo de métricas financeiras como Índice Sharpe, Sortino, Treynor, etc.
- `visualization.py`: Funções para exibição de gráficos e tabelas interativas.

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositório>
cd <nome-do-repositório>
```

2. Crie e ative um ambiente virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
```

3. Instale as dependências necessárias:
```bash
pip install streamlit yfinance pandas numpy matplotlib scipy plotly seaborn
```

## Como Usar

1. Inicie a aplicação Streamlit:
```bash
streamlit run app.py
```

2. Acesse a interface web pelo seu navegador (geralmente http://localhost:8501)

3. Configure seu portfólio:
   - Defina o valor do investimento
   - Selecione o período para análise de dados históricos
   - Escolha ações da lista padrão ou adicione tickers personalizados
   - Ajuste os parâmetros do algoritmo (tamanho da população, gerações, taxa de mutação)

4. Clique em "🚀 Optimize Portfolio" para iniciar o processo de otimização

## Observações

- Dados históricos não garantem performance futura
- O processo de otimização utiliza o Índice Sharpe para retornos ajustados ao risco
- A taxa livre de risco padrão é 2%, mas pode ser ajustada
- A aplicação utiliza preços de fechamento ajustados para os cálculos

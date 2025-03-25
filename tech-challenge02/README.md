# Otimizador de Portfólio com Algoritmo Genético

Uma aplicação web em Streamlit que utiliza algoritmos genéticos para otimizar a alocação de portfólio de investimentos baseado no Índice Sharpe. Esta ferramenta ajuda investidores a encontrar pesos ótimos para o portfólio através da análise de dados históricos de ações e maximização dos retornos ajustados ao risco.

## Funcionalidades

- Otimização de Portfólio:
  - Algoritmos genéticos para encontrar a melhor alocação de ativos.
  - Suporte a multiobjetivo (retorno e risco).
- Configuração Personalizável:
  - Valor do investimento, período de análise e seleção de ações.
  - Ajuste de parâmetros do algoritmo (população, gerações, taxa de mutação, etc.).
  - Suporte a diferentes estratégias de inicialização e métodos de seleção.
- Visualizações Interativas:
  - Gráficos de progresso da otimização.
  - Alocação do portfólio em gráficos de pizza.
  - Evolução do Pareto Front para análises multiobjetivo.
  - Gráficos de correlação e fronteira eficiente.
- Métricas Detalhadas:
  - Retorno esperado, volatilidade e Índice Sharpe.
  - Projeções de investimento para diferentes horizontes temporais.
  - Análise de risco com Value at Risk (VaR) e Conditional VaR (CVaR).
- Exportação de Resultados:
  - Baixe os resultados do portfólio em formato CSV.

## Estrutura do Projeto

O projeto foi modularizado para facilitar a manutenção e escalabilidade:

- `app.py`: Arquivo principal que gerencia a interface do usuário com Streamlit.
- `src/metrics/performance.py`: Cálculo de métricas de desempenho como Índice Sharpe, Sortino e Treynor.
- `src/metrics/risk.py`: Cálculo de métricas de risco como Value at Risk (VaR) e Conditional VaR (CVaR).
- `src/data/loader.py`: Funções para download e tratamento de dados históricos de ações.
- `src/models/genetic_algorithm.py`: Implementação detalhada do algoritmo genético.
- `src/visualization/plots.py`: Funções para exibição de gráficos e tabelas interativas.
- `config.py`: Configurações globais e parâmetros padrão do projeto.

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
pip install -r requirements.txt
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

4. Clique em "🚀 Otimizar Portfólio" para iniciar o processo de otimização

## Parâmetros do Algoritmo

- **População**: Número de indivíduos na população.
- **Gerações**: Número de iterações do algoritmo.
- **Taxa de Mutação**: Probabilidade de mutação em cada indivíduo.
- **Taxa Livre de Risco**: Taxa de retorno sem risco usada no cálculo do índice Sharpe.
- **Método de Seleção**: Estratégia para selecionar pais (ex.: torneio, roleta, elitismo).
- **Método de Crossover**: Estratégia para combinar pais (ex.: uniforme, ponto único, aritmético).
- **Distribuição de Mutação**: Tipo de distribuição para mutação (ex.: normal, uniforme).
- **Estratégia de Inicialização**: Métodos para inicializar os pesos do portfólio (ex.: aleatória, uniforme, diversificada).

---

## Principais Funções

### `create_individual(size, strategy="random", returns=None)`
Cria um indivíduo (pesos do portfólio) com base na estratégia especificada.

**Parâmetros**:
- `size` (int): Número de ativos no portfólio.
- `strategy` (str): Estratégia de inicialização ("random", "uniform", "return_based", "volatility_inverse").
- `returns` (pd.DataFrame, opcional): Retornos históricos dos ativos (necessário para algumas estratégias).

**Retorna**:
- `np.ndarray`: Pesos normalizados do portfólio.

---

### `evaluate_population(population, returns, cov_matrix, risk_free_rate, metric=None, market_returns=None, multiobjective=False)`
Avalia a população de portfólios com base em métricas de desempenho.

**Parâmetros**:
- `population` (list): Lista de indivíduos (pesos do portfólio).
- `returns` (pd.DataFrame): Retornos históricos dos ativos.
- `cov_matrix` (pd.DataFrame): Matriz de covariância dos retornos.
- `risk_free_rate` (float): Taxa livre de risco.
- `metric` (str, opcional): Métrica de avaliação ("sharpe", "sortino", "treynor", "var").
- `market_returns` (pd.Series, opcional): Retornos do mercado (necessário para algumas métricas).
- `multiobjective` (bool): Se `True`, avalia retorno e risco como objetivos separados.

**Retorna**:
- `list`: Lista de scores de fitness para cada indivíduo.

---

### `select_pareto_front(population, fitness_scores)`
Seleciona o Pareto Front (conjunto de soluções não dominadas).

**Parâmetros**:
- `population` (list): Lista de indivíduos (pesos do portfólio).
- `fitness_scores` (list): Lista de scores de fitness (retorno e risco).

**Retorna**:
- `list`: Lista de indivíduos e seus scores no Pareto Front.

---

### `optimize_portfolio(...)`
Função principal que executa o algoritmo genético para otimização do portfólio.

**Parâmetros**:
- `selected_tickers` (list): Lista de tickers selecionados.
- `start_date` (str): Data inicial para análise.
- `end_date` (str): Data final para análise.
- `investment` (float): Valor total do investimento.
- `population_size` (int): Tamanho da população.
- `num_generations` (int): Número de gerações.
- `mutation_rate` (float): Taxa de mutação.
- `risk_free_rate` (float): Taxa livre de risco.
- `min_weight` (float): Peso mínimo permitido para cada ativo.
- `max_weight` (float): Peso máximo permitido para cada ativo.
- Outros parâmetros para personalização do algoritmo.

## Observações

- Dados históricos não garantem performance futura
- O processo de otimização utiliza o Índice Sharpe para retornos ajustados ao risco
- A taxa livre de risco padrão é 2%, mas pode ser ajustada
- A aplicação utiliza preços de fechamento ajustados para os cálculos

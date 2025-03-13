# Otimizador de Portfólio com Algoritmo Genético

Uma aplicação web em Streamlit que utiliza algoritmos genéticos para otimizar a alocação de portfólio de investimentos baseado no Índice Sharpe. Esta ferramenta ajuda investidores a encontrar pesos ótimos para o portfólio através da análise de dados históricos de ações e maximização dos retornos ajustados ao risco.

## Funcionalidades

- Otimização de portfólio em tempo real usando algoritmos genéticos
- Seleção interativa de parâmetros:
  - Personalização do valor do investimento
  - Seleção do período de análise
  - Seleção flexível de ações (padrão + tickers personalizados)
  - Parâmetros ajustáveis do algoritmo
- Visualização em tempo real do processo de otimização
- Métricas detalhadas do portfólio:
  - Retornos esperados
  - Análise de volatilidade
  - Cálculos do Índice Sharpe
- Projeções de investimento para diferentes horizontes temporais
- Visualização em tempo real da alocação do portfólio

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
pip install streamlit yfinance pandas numpy matplotlib
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

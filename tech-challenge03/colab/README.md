# 🧠 Fine-Tuning com LLaMA 3.2 3B para Geração de Descrições de Produtos

## 📝 Descrição Geral

Este repositório é parte do Tech Challenge e demonstra como realizar o fine-tuning de um modelo foundation (LLaMA 3.2 3B Instruct) usando o dataset "The AmazonTitles-1.3MM". O objetivo é treinar o modelo para responder a perguntas sobre produtos, gerando suas descrições com base nos títulos presentes no dataset.

## 📂 Estrutura do Projeto

```
TechChallenge3/
├── trn.json               # Dataset de treinamento
├── tst.json               # Dataset de teste
├── output/
│   └── train_prepared.jsonl / test_prepared.jsonl
├── model/                 # Diretório onde o modelo fine-tunado é salvo
└── notebook.ipynb         # Notebook com o treinamento e inferência
```

## 📌 Objetivos

* Realizar o fine-tuning de um modelo LLaMA com base em perguntas sobre produtos
* Gerar descrições realistas e contextuais com base no título
* Demonstrar o processo completo: preparação, treinamento e inferência

## ⚙️ Requisitos

* Google Colab
* GPU (idealmente com suporte a 4-bit quantization)

## 🧪 Execução

Abra o notebook `notebook.ipynb` no Google Colab e siga as etapas:

1. **Montar o Google Drive** (armazenamento de dados e modelo)
2. **Instalar dependências**
3. **Preparar os dados** (`title` → prompt; `content` → resposta)
4. **Carregar modelo** (`unsloth/Llama-3.2-3B-Instruct`)
5. **Executar fine-tuning** com LoRA e 4-bit
6. **Testar inferência** com exemplos reais
7. **Salvar modelo final** no diretório `model/`

## 💬 Exemplo de Uso

**Entrada**:

```text
Qual é a descrição do produto com o título: Girls Ballet Tutu Neon Pink?
```

**Saída gerada pelo modelo**:

```text
Este tutu de balé rosa neon é perfeito para meninas que adoram dançar. Feito com camadas de tule macio, proporciona conforto e estilo para qualquer apresentação.
```

## 🧠 Modelo Utilizado

* `unsloth/Llama-3.2-3B-Instruct`
* Fine-tuning com LoRA
* Quantização em 4-bit

## 🗃️ Dataset

* [The AmazonTitles-1.3MM](https://drive.google.com/file/d/12zH4mL2RX8iSvH0VCNnd3QxO4DzuHWnK/view)
* Campos utilizados: `title` (título do produto) e `content` (descrição)

## 🧾 Parâmetros de Treinamento

* batch: 2 (acumulado 4)
* max steps: 60
* learning rate: 2e-4
* warmup steps: 5
* weight decay: 0.01

## 🎥 Demonstração

Grave um vídeo apresentando:

* Seu notebook
* Uma pergunta feita ao modelo
* A resposta gerada

## ✅ Resultado Final

O modelo consegue responder perguntas como:

> "Qual é a descrição do produto com o título: X?"

Utilizando os aprendizados do fine-tuning para gerar descrições coerentes com o conteúdo de treinamento.

## 📬 Contato

Para dúvidas ou sugestões, entre em contato via GitHub ou email pessoal.

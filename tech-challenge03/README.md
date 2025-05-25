# 🧠 Fine-Tuning com Unsloth + TinyLlama no Dataset AmazonTitles-1.3MM

## 📘 Visão Geral

Este projeto foi desenvolvido como parte de um **Tech Challenge**, com o objetivo de realizar **fine-tuning de um modelo LLM** usando o dataset **AmazonTitles-1.3MM**, contendo títulos e descrições de produtos. O modelo aprende a gerar descrições de produtos com base apenas em seus títulos.

Utilizamos o modelo `unsloth/tinyllama-bnb-4bit` pela leveza e performance, com **LoRA** e **4-bit quantization**, utilizando **Google Colab** com GPU.

---

## 🧾 Sumário

* [Modelo Utilizado](#modelo-utilizado)
* [Dataset](#dataset)
* [Fluxo de Treinamento](#fluxo-de-treinamento)
* [Exemplos](#exemplos)
* [Resultados](#resultados)
* [Conclusão](#conclusão)

---

## 💡 Modelo Utilizado

* **Modelo:** `unsloth/tinyllama-bnb-4bit`
* **Framework:** [Unsloth](https://github.com/unslothai/unsloth)
* **Técnica:** LoRA
* **Formato:** 4-bit quantization (bnb)

---

## 📂 Dataset

Utilizado o dataset **AmazonTitles-1.3MM** com campos:

```json
{
  "uid": "0000032069", 
  "title": "Adult Ballet Tutu Cheetah Pink", 
  "content": "..."
}
```

Após o pré-processamento, o dataset foi transformado no formato:

```json
{
  "instruction": "Responda com a descrição do produto baseado no título fornecido.",
  "input": "Adult Ballet Tutu Cheetah Pink",
  "output": "Uma descrição relevante do produto."
}
```

---

## ⚙️ Fluxo de Treinamento

### 📦 1. Montagem do Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 📥 2. Instalação de Dependências

```python
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps xformers "trl<0.9.0" peft accelerate bitsandbytes
!pip install transformers datasets trl
```

### 🔄 3. Preparação dos Dados

```python
def prepare_data(input_path, output_path):
    ...
    # Gera arquivos train_data.jsonl e test_data.jsonl
```

---

### 🧠 4. Carregamento do Modelo

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/tinyllama-bnb-4bit",
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)
```

### 🔧 5. Aplicação do LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

### ✅ 6. Preparar para Treinamento

```python
model = FastLanguageModel.for_training(model)
```

---

### 🔍 7. Teste Antes do Fine-Tuning

```python
def gerar_resposta_pretreino(produto):
    ...
print(gerar_resposta_pretreino("Mog's Family of Cats"))
```

---

### 📚 8. Carregamento do Dataset

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files=".../train_data.jsonl", split="train")
dataset = dataset.shuffle(seed=42).select(range(50000))
```

---

### 🚀 9. Treinamento com `SFTTrainer`

```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="output",
    tokenizer=tokenizer,
    args=TrainingArguments(
        output_dir="model",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        fp16=True,
        ...
    )
)
trainer.train()
```

---

### 💾 10. Salvamento do Modelo

```python
model.save_pretrained("/content/drive/MyDrive/TechChallenge3/model/")
tokenizer.save_pretrained("/content/drive/MyDrive/TechChallenge3/model/")
```

---

## 🧪 Exemplos

Após o fine-tuning:

```python
def gerar_resposta(produto):
    prompt = f"### Instrução:\nDescreva o seguinte produto:\n### Produto:\n{produto}\n### Resposta:"
    ...
```

### 🔍 Resultados:

```python
gerar_resposta("Mog's Family of Cats")
gerar_resposta("Why Don't They Just Quit? DVD Roundtable Discussion")
```

---

## 📊 Resultados

| Métrica               | Antes do Treino | Após o Treino |
| --------------------- | --------------- | ------------- |
| Coerência             | Baixa           | Alta          |
| Relevância Contextual | Fraca           | Boa           |
| Aderência à Instrução | Irregular       | Consistente   |

---

## ✅ Conclusão

* O uso do **TinyLlama com Unsloth e LoRA** permitiu treinar um modelo leve e eficaz diretamente no **Google Colab**.
* O modelo conseguiu **aprender padrões do dataset** e responder com descrições contextualizadas de produtos.
* É possível expandir o projeto com mais épocas, validação e métricas automáticas (BLEU, ROUGE, etc).

---

## 📁 Estrutura de Arquivos

```
TechChallenge3/
├── trn.json
├── tst.json
├── output/
│   ├── train_data.jsonl
│   └── test_data.jsonl
├── model/
│   ├── adapter_model.bin
│   └── tokenizer_config.json
```

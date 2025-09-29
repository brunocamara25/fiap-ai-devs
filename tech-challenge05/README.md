# Modelagem de Ameaças em Diagramas de Arquitetura com IA

MVP que detecta componentes de arquitetura em **imagens de diagramas** (cloud/sistemas) e gera um **relatório STRIDE** automático. O projeto inclui:

* pipeline para **organizar ícones**,
* **gerar dados sintéticos**,
* **normalizar datasets VOC → YOLO**,
* **unificar** dataset real + sintético,
* **treinar YOLO**,
* demo em **Streamlit** com OCR e relatório STRIDE.

---

## 1) Visão Geral

**Objetivo**
Interpretar imagens de diagramas, identificar componentes (API, VM, DB, etc.) e produzir um relatório de ameaças **STRIDE** com contramedidas sugeridas.

**Abordagem**

1. Detecção supervisionada (YOLO) dos elementos do diagrama.
2. OCR para textos de apoio.
3. Análise STRIDE (LLM + heurísticas).
4. Exportação de relatório.

---

## 2) Pipeline (passo a passo)

1. **Classificar/organizar ícones por classe**
2. **Gerar dataset sintético (imagens + labels YOLO)**
3. **Normalizar dataset real VOC → YOLO** (mesmo vocabulário)
4. **Mesclar real + sintético**
5. **Treinar YOLO** (MPS/Colab)
6. **Rodar a demo (Streamlit) e gerar relatório STRIDE**

---

## 3) Estrutura do Repositório

```
├── config/
│   ├── icons.yml            # regras/LLM para classificar ícones
│   └── dataset.yml          # normalização VOC→YOLO e merge
├── data/
│   ├── icons/               # ícones de entrada (AWS/Azure/GCP…)
│   └── output/              # saídas do classificador de ícones
├── datasets/
│   ├── synthetic/           # dataset sintético gerado
│   ├── real/                # dataset real normalizado (VOC→YOLO)
│   └── merged/              # real + sintético unificado p/ treino
├── demo/
│   └── streamlit_app.py     # demo interativa e relatório STRIDE
├── tools/
│   ├── auto_build_classes.py  # organiza ícones em classes
│   ├── generate_synthetic.py  # gera imagens/labels sintéticos
│   ├── dataset_unify.py       # VOC→YOLO + merge de datasets
│   ├── shrink_images.py       # otimização/resize em lote
│   ├── train_yolo.py          # treino (opcional) / exemplos
│   └── stride_json_utils.py   # utilidades STRIDE
└── README.md
```

---

## 4) Ambiente

```bash
python -m venv venv && source venv/bin/activate   # (Linux/Mac)
# ou: .\venv\Scripts\activate (Windows)

pip install -r requirements.txt
# extras opcionais:
# - cairosvg (SVG)   -> pip install cairosvg
# - ollama (LLM local)-> https://ollama.com (e baixe o modelo "llama3")
```

---

## 5) Passo 1 — Classificação de Ícones

Organize os ícones por pastas/classes usando heurísticas e, opcionalmente, LLM.

**Config (exemplo `config/icons.yml`):**

```yaml
paths:
  icons_root: data/icons/
  out: data/output/
  copy_files: false

classes:
  closed_set: [api_gateway, vm_compute, relational_db, object_storage, ...]
  include_extras: true

heuristics:
  token2class:
    db, database: relational_db
    api, apigateway, apim: api_gateway
    vm, ec2, compute_engine, virtual_machine: vm_compute

llm:
  enabled: true
  model: llama3
  temperature: 0.0
  batch_size: 40

outputs:
  emit: [classes_txt, icon_map_csv, class_stats_json, folder_unify_json, unknown_icons_csv]
```

**Rodar:**

```bash
python tools/auto_build_classes.py --config config/icons.yml
```

**Saídas principais**:

* `outputs/classes.yaml` (vocabulário canônico)
* `outputs/icons/<classe>/...` (ícones organizados)

---

## 6) Passo 2 — Geração de Dados Sintéticos

Gera imagens de diagramas + labels YOLO com ruídos/augmentations leves.

**Exemplo:**

```bash
python tools/generate_synthetic.py \
  --icons outputs/icons \
  --classes_file outputs/classes.yaml \
  --out datasets/synthetic \
  --num 2000 \
  --canvas 1600x1000 \
  --min_nodes 8 --max_nodes 20 \
  --svg_px 256
```

**Saídas**: `images/`, `labels/`, `classes.yaml`, `data.yaml`, `splits/`.

---

## 7) Passo 3 — Normalizar Dataset Real (VOC → YOLO)

Base usada (VOC): **Software Architecture Dataset**
Kaggle: [https://www.kaggle.com/datasets/carlosrian/software-architecture-dataset/data](https://www.kaggle.com/datasets/carlosrian/software-architecture-dataset/data)

Use o mesmo vocabulário do sintético e um **config** unificado.

**Config (exemplo `config/dataset.yml`):**

```yaml
paths:
  xml_root: ../../dataset                  # VOC XMLs
  out_root: ../../datasets/real            # saída YOLO
  classes: outputs/classes.yaml            # MESMO vocabulário
  img_roots: [../../dataset]
  copy_images: true
  resize_long: 1280                        # reduz imagens grandes
  val_ratio: 0.1

llm:
  enabled: true
  model: "llama3"
  batch: 60
  temperature: 0.2

matching:
  auto_fuzzy: true
  fuzzy_cutoff: 0.90
  use_llm: true

normalization:
  # mapeamentos explícitos (reduz 'unknowns')
  class_map:
    aws_simple_queue_service_queue: event_topic_pubsub
    aws_simple_notification_service_topic: event_topic_pubsub
    gcp_pubsub: event_topic_pubsub
    aws_amazon_simple_queue_service: event_topic_pubsub
    aws_amazon_simple_notification_service: event_topic_pubsub
    aws_amazon_simple_storage_service: object_storage
    aws_simple_storage_service_object: object_storage
    azure_data_factories: etl_data_factory
    azure_event_hubs: event_topic_pubsub
    # ignore ou mapeie
    aws_backup: ignore
    sass_services: ignore

  # sinônimos frequentes → classe canônica
  synonyms:
    queue: event_topic_pubsub
    notification: event_topic_pubsub
    pubsub: event_topic_pubsub
    event_hubs: event_topic_pubsub
    service_bus: event_topic_pubsub
    developer_portal: api_gateway
    compute_engine: vm_compute
    ec2: vm_compute
    virtual_machine: vm_compute
    storage_account: object_storage
    bucket: object_storage
    s3: object_storage
    sql: relational_db
    cloud_sql: relational_db
    rds: relational_db
    bigquery: data_warehouse
    synapse: data_warehouse
    redshift: data_warehouse
    lambda: serverless_functions
    cloud_functions: serverless_functions
    gke: kubernetes
    eks: kubernetes
    aks: kubernetes

heuristics:
  token2class:
    - [["api_gateway","apim","apigee","developer_portal","api"], api_gateway]
    - [["cdn","cloudfront","front_door"], cdn]
    - [["gke","eks","aks","kubernetes"], kubernetes]
    - [["ecs","container","cloud_run"], container_service]
    - [["lambda","functions","function_app","cloud_functions"], serverless_functions]
    - [["rds","sql_server","cloud_sql","postgres","mysql","aurora","sql"], relational_db]
    - [["dynamodb","cosmos_db","solr"], nosql_db]
    - [["s3","storage_account","cloud_storage","bucket","blob","object_storage"], object_storage]
    - [["efs","filestore","file_share","file_system"], file_storage]
    - [["ebs","disk","block_store"], block_storage]
    - [["bigquery","synapse","redshift","data_warehouse","databricks"], data_warehouse]
    - [["datafactory","dataflow","dataproc"], etl_data_factory]
    - [["sns","sqs","queue","topic","pubsub","event_hubs","service_bus"], event_topic_pubsub]
    - [["cloudwatch","application_insights","monitor"], monitoring]
    - [["logging","cloud_trail","audit"], logging]
    - [["iam","identity","entra","user"], iam_identity]
    - [["kms","key_vault","secret_manager","certificate_manager"], kms_key_vault]
    - [["waf","firewall"], waf]
    - [["security","defender","guardduty","inspector","network_security_group"], security]
    - [["vpc","vnet","virtual_network","subnet","resource_group","region","cloud"], vpc_vnet]
    - [["dns","route_53","dns_zone","cloud_dns"], dns]
    - [["alb","nlb","load_balancer","traffic_director","autoscaling"], load_balancer]
    - [["vpn"], vpn_gateway]
    - [["vm","ec2","compute_engine","virtual_machine"], vm_compute]
```

**Rodar:**

```bash
python tools/dataset_unify.py from_voc --config config/dataset.yml
```

**Saídas**: `datasets/real/{images,labels,splits,data.yaml,reports}`

> Dica: se ainda houver muitos *unknowns*, adicione-os em `normalization.class_map` e rode novamente.

---

## 8) Passo 4 — Merge (Real + Sintético)

**No mesmo `config/dataset.yml` adicione:**

```yaml
merge:
  sources:
    - ./datasets/real
    - ./datasets/synthetic     # ou outro(s)
```

**Rodar:**

```bash
python tools/dataset_unify.py merge --config config/dataset.yml
# saída em ./datasets/merged
```

---

## 9) Passo 5 — Treinamento YOLO

### (A) Treino rápido no Apple Silicon (MPS)

```bash
python tools/train_yolo.py \
  --data datasets/merged/data.yaml \
  --runs_dir runs \
  --run_name yolo_syn_real_mps_s \
  --base_model yolov8s.pt \
  --imgsz 640 \
  --epochs 60 \
  --batch 12 \
  --device mps \
  --workers 0
```

### (B) Colab (GPU T4) — exemplo com augmentations leves

```python
from ultralytics import YOLO

DATA_YAML = "/content/drive/MyDrive/tech-challenge/datasets/merged/data.yaml"
runs_dir  = "/content/drive/MyDrive/tech-challenge/runs"
run_name  = "yolo_new_s"

model = YOLO("yolov8s.pt")
model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=832,
    batch=-1,
    device=0,
    workers=4,
    amp=True,
    project=runs_dir, name=run_name,
    mosaic=0, mixup=0, copy_paste=0,
    fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=3, translate=0.06, scale=0.9, perspective=0.0,
    val=False, save=True, save_period=5,
)

# validação ao final
metrics = model.val(data=DATA_YAML, imgsz=832, device=0, conf=0.25, iou=0.6, plots=True)
print(metrics.results_dict)
```

> **Tip**: faça um fine-tuning curto (20–30 épocas) partindo de `last.pt` com LR menor para aparar erros em classes específicas.

---

## 10) Passo 6 — Demo (Streamlit) + STRIDE

Pré-requisitos:

* Modelo YOLO em `demo/models/best.pt`
* `demo/data.yaml` compatível (mesmas classes)
* Tesseract instalado (OCR):

  * macOS: `brew install tesseract`
  * Ubuntu: `sudo apt-get install tesseract-ocr`
  * Windows: baixe do site oficial

**Rodar:**

```bash
streamlit run demo/streamlit_app.py
```

Fluxo da demo:

1. Upload do diagrama (PNG/JPG).
2. YOLO detecta componentes.
3. OCR extrai textos.
4. Analisador STRIDE (LLM/heurísticas) gera JSON de ameaças + contramedidas.
5. Revisar, editar e **exportar PDF**.

---

## 11) Datasets

* **VOC → YOLO (real)**: *Software Architecture Dataset*
  [https://www.kaggle.com/datasets/carlosrian/software-architecture-dataset/data](https://www.kaggle.com/datasets/carlosrian/software-architecture-dataset/data)
* **Sintético**: gerado a partir de ícones organizados por classe (seções 5–6).

---

## 12) Resultados (exemplo)

* mAP50-95 ≈ **0.90** em validação com YOLOv8s (dataset unificado).
* Classes mais desafiadoras melhoram com **lote sintético focado** e **fine-tuning**.

---

## 13) Dicas e Solução de Problemas

* **Muitos rótulos desconhecidos (VOC→YOLO)**:
  amplie `normalization.class_map` e `synonyms` no `config/dataset.yml`, reexecute `from_voc`.
* **Imagens muito grandes**: use `resize_long: 1280` na normalização e/ou `tools/shrink_images.py`.
* **LLM não mapeia**: confirme se o Ollama está rodando e se `llm.enabled: true`. Sempre prefira **`class_map` explícito** para rótulos frequentes.
* **Desbalanceamento**: gere sintéticos focados (mais instâncias) das classes fracas.
* **Diferença de vocabulário**: garanta que **todas as etapas** usam **o mesmo** `outputs/classes.yaml`.

---

## 14) Comandos Resumo

```bash
# 1) Ícones → classes
python tools/auto_build_classes.py --config config/icons.yml

# 2) Sintético
python tools/generate_synthetic.py --icons outputs/icons --classes_file outputs/classes.yaml --out datasets/synthetic --num 2000

# 3) Real (VOC→YOLO)
python tools/dataset_unify.py from_voc --config config/dataset.yml

# 4) Merge
python tools/dataset_unify.py merge --config config/dataset.yml

# 5) Treino (exemplo MPS)
python tools/train_yolo.py --data datasets/merged/data.yaml --device mps
```

---

## 15) Licenças / Observações

* Respeite as licenças dos ícones e datasets utilizados.
* Este repositório é um MVP educacional/experimental e pode exigir ajustes para produção (otimizações, testes, segurança operacional).

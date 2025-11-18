# Federated Learning untuk Fine-tuning GPT-2 dengan LoRA

Sistem federated learning lengkap untuk fine-tuning model GPT-2 menggunakan teknik LoRA (Low-Rank Adaptation).

## 📁 Struktur File

```
fed_learning/
├── __init__.py              # Module initialization
├── config.py                # Konfigurasi sistem (CONFIGURABLE)
├── dataset.py               # Dataset handling & distribution
├── client.py                # Client implementation
├── server.py                # Server implementation
└── federated_learning.py    # Main orchestrator
```

## 🎯 Komponen Utama

### 1. **Config** (`config.py`)

File konfigurasi lengkap yang dapat disesuaikan:

- **ModelConfig**: Konfigurasi model (GPT-2, max_length, dll)
- **LoRAConfig**: Parameter LoRA (rank, alpha, target modules, dropout)
- **TrainingConfig**: Hyperparameter training (epochs, batch_size, learning_rate, dll)
- **FederatedConfig**: Parameter federated learning (num_clients, num_rounds, data_distribution, dll)
- **DataConfig**: Konfigurasi dataset (path, columns, splits)
- **ExperimentConfig**: Konfigurasi keseluruhan eksperimen

### 2. **Dataset** (`dataset.py`)

Mengelola data dan distribusi ke clients:

- Load data dari CSV
- Tokenisasi dengan AutoTokenizer
- Distribusi data:
  - **IID**: Random distribution (Independent and Identically Distributed)
  - **Non-IID**: Heterogeneous distribution berdasarkan panjang teks
  - **By Author**: Natural non-IID, setiap client = author berbeda

### 3. **Client** (`client.py`)

Implementasi client untuk training lokal:

- Setup model GPT-2 dengan LoRA adapters
- Local training dengan Hugging Face Trainer
- Get/set model weights (hanya LoRA adapters)
- Evaluation dan text generation
- Training history tracking

### 4. **Server** (`server.py`)

Server untuk koordinasi dan agregasi:

- Initialize global model
- Client selection per round
- Model aggregation (FedAvg)
- Global model evaluation
- Model saving dan metrics tracking

### 5. **Federated Learning** (`federated_learning.py`)

Main orchestrator:

- Setup seluruh sistem
- Koordinasi training rounds
- Tracking metrics dan history
- Visualization (plotting)
- Text generation testing

## 🚀 Cara Penggunaan

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Jalankan dengan Default Configuration

```python
python fed_learning/federated_learning.py
```

### 3. Kustomisasi Configuration

#### Opsi A: Edit langsung di code

```python
from fed_learning import FederatedLearning, get_default_config

# Get default config
config = get_default_config()

# Customize
config.federated.num_rounds = 10
config.federated.num_clients = 5
config.federated.clients_per_round = 3
config.federated.data_distribution = "by_author"  # "iid", "non_iid", "by_author"
config.training.num_epochs = 3
config.training.batch_size = 8
config.training.learning_rate = 5e-5

# Save config
config.save("my_config.json")

# Run
fed_learning = FederatedLearning(config)
fed_learning.setup()
fed_learning.train()
```

#### Opsi B: Load dari JSON file

```python
from fed_learning import FederatedLearning, ExperimentConfig

# Load config
config = ExperimentConfig.load("my_config.json")

# Run
fed_learning = FederatedLearning(config)
fed_learning.setup()
fed_learning.train()
```

### 4. Kustomisasi Lanjutan

```python
from fed_learning.config import (
    ExperimentConfig,
    ModelConfig,
    LoRAConfig,
    TrainingConfig,
    FederatedConfig,
    DataConfig
)

# Create custom config
config = ExperimentConfig(
    experiment_name="my_experiment",
    output_dir="./my_results",
    seed=42,
    device="cuda",

    model=ModelConfig(
        model_name="gpt2-medium",  # Atau "gpt2-large"
        max_length=256,
    ),

    lora=LoRAConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.1,
    ),

    training=TrainingConfig(
        num_epochs=5,
        batch_size=4,
        learning_rate=3e-5,
        warmup_steps=100,
    ),

    federated=FederatedConfig(
        num_clients=10,
        num_rounds=20,
        clients_per_round=5,
        data_distribution="by_author",
        aggregation_method="fedavg",
    ),

    data=DataConfig(
        dataset_path="./twcs/twcs.csv",
        text_column="text",
        author_column="author_id",
        filter_inbound_only=True,
    )
)

# Save
config.save("custom_config.json")
```

## 📊 Output

Sistem akan menghasilkan:

1. **Models**: `fed_results/global_model/round_X/`

   - LoRA adapter weights
   - Model configuration

2. **Metrics**: `fed_results/`

   - `training_metrics.json`: Detailed metrics per round
   - `global_metrics.json`: Global model performance
   - `training_progress.png`: Visualization plots

3. **Logs**: `fed_logs/`
   - Training logs per client

## 🔧 Parameter Penting

### LoRA Parameters

- `r`: Rank of LoRA matrices (default: 8, higher = more parameters)
- `lora_alpha`: Scaling factor (default: 16)
- `target_modules`: Which layers to apply LoRA (GPT-2: ["c_attn", "c_proj"])

### Federated Parameters

- `num_clients`: Total number of clients
- `num_rounds`: Number of federated rounds
- `clients_per_round`: How many clients participate per round
- `data_distribution`: "iid", "non_iid", or "by_author"
- `aggregation_method`: "fedavg" (weighted average)

### Training Parameters

- `num_epochs`: Local epochs per round
- `batch_size`: Batch size for training
- `learning_rate`: Learning rate
- `gradient_accumulation_steps`: For larger effective batch size

## 📈 Monitoring

Training progress ditampilkan di console:

```
====================================================================
FEDERATED LEARNING ROUND 1/10
====================================================================

Selected clients: [0, 2, 4]

----------------------------------------------------------------------
CLIENT TRAINING
----------------------------------------------------------------------
[Client 0] Training for 3 epochs...
[Client 0] Training complete! Loss: 2.1234

[Client 2] Training for 3 epochs...
[Client 2] Training complete! Loss: 2.0987

...

----------------------------------------------------------------------
MODEL AGGREGATION
----------------------------------------------------------------------
Aggregating models using FedAvg...
Global model updated successfully!

----------------------------------------------------------------------
GLOBAL MODEL EVALUATION
----------------------------------------------------------------------
Global Evaluation Loss: 2.0543
Global Perplexity: 7.8012
====================================================================
```

## 🎯 Contoh Use Cases

### 1. IID Distribution (Balanced)

```python
config.federated.data_distribution = "iid"
config.federated.num_clients = 5
```

### 2. Non-IID Distribution (Heterogeneous)

```python
config.federated.data_distribution = "non_iid"
config.federated.num_clients = 5
```

### 3. By Author (Real-world scenario)

```python
config.federated.data_distribution = "by_author"
config.federated.num_clients = 10  # Top 10 authors
config.federated.min_samples_per_client = 100
```

## 🧪 Testing Generation

```python
fed_learning.test_generation([
    "Hello, I need help with",
    "My order is",
    "Can you tell me about",
])
```

## 📝 Notes

- Gunakan GPU untuk training lebih cepat: `config.device = "cuda"`
- Untuk Mac dengan M1/M2: `config.device = "mps"`
- FP16 training untuk efisiensi: `config.training.fp16 = True`
- Save model setiap N rounds di `federated_learning.py` line 243

## 🔍 Troubleshooting

1. **Out of Memory**: Kurangi `batch_size` atau `max_length`
2. **Slow Training**: Enable FP16, kurangi `num_epochs`
3. **Not enough authors**: Turunkan `min_samples_per_client`
4. **Poor convergence**: Tingkatkan `learning_rate` atau `num_rounds`

## 📚 References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Federated Learning](https://arxiv.org/abs/1602.05629)
- [Hugging Face PEFT](https://github.com/huggingface/peft)

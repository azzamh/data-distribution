# 🚀 Sistem Federated Learning untuk Fine-tuning GPT-2 dengan LoRA

## 📋 Ringkasan

Sistem federated learning lengkap dan configurable untuk fine-tuning model GPT-2 menggunakan teknik LoRA (Low-Rank Adaptation). Sistem ini memungkinkan training terdistribusi dengan berbagai strategi distribusi data (IID, Non-IID, by Author).

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────┐
│                   FederatedLearning                     │
│                    (Orchestrator)                       │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼───┐   ┌───▼────┐   ┌──▼─────┐
   │ Server │   │Dataset │   │ Config │
   └────┬───┘   └───┬────┘   └────────┘
        │           │
        │      ┌────▼────┐
        │      │ Client  │
        │      │  (1-N)  │
        │      └─────────┘
        │
   ┌────▼──────────────────┐
   │   Global Model        │
   │  (GPT-2 + LoRA)      │
   └───────────────────────┘
```

## 📦 Komponen Sistem

### 1. **config.py** - Configuration Management

- `ExperimentConfig`: Master configuration class
- `ModelConfig`: Model settings (GPT-2 variant, max_length)
- `LoRAConfig`: LoRA parameters (rank, alpha, target modules)
- `TrainingConfig`: Training hyperparameters
- `FederatedConfig`: Federated learning settings
- `DataConfig`: Dataset configuration

**Features:**

- ✅ Fully configurable via Python objects
- ✅ Save/Load dari JSON
- ✅ Validation dan default values
- ✅ Pretty printing untuk debugging

### 2. **dataset.py** - Data Management

- `FederatedDataset`: Main dataset handler

**Features:**

- ✅ Load dari CSV (Twitter Customer Support dataset)
- ✅ Automatic tokenization dengan GPT-2 tokenizer
- ✅ 3 strategi distribusi data:
  - **IID**: Random uniform distribution
  - **Non-IID**: Heterogeneous (by text length)
  - **By Author**: Natural non-IID (setiap client = author berbeda)
- ✅ Automatic train/val/test split
- ✅ Data filtering (min length, inbound only, dll)

### 3. **client.py** - Client Implementation

- `FederatedClient`: Local training client

**Features:**

- ✅ Setup GPT-2 dengan LoRA adapters
- ✅ Local training dengan Hugging Face Trainer
- ✅ Get/Set LoRA weights (hanya trainable params)
- ✅ Model evaluation
- ✅ Text generation
- ✅ Training history tracking
- ✅ Per-round checkpointing

### 4. **server.py** - Server Implementation

- `FederatedServer`: Coordination server

**Features:**

- ✅ Initialize global model
- ✅ Random client selection per round
- ✅ FedAvg aggregation (weighted average by dataset size)
- ✅ Global model evaluation
- ✅ Model checkpointing
- ✅ Metrics tracking
- ✅ Text generation testing

### 5. **federated_learning.py** - Main Orchestrator

- `FederatedLearning`: Main coordinator class

**Features:**

- ✅ End-to-end workflow orchestration
- ✅ Multi-round training automation
- ✅ Comprehensive logging
- ✅ Metrics collection & visualization
- ✅ Progress tracking
- ✅ Automatic plotting (loss, perplexity)
- ✅ Text generation testing

## 🎯 Workflow Federated Learning

```
1. SETUP
   ├── Load Configuration
   ├── Initialize Dataset Manager
   │   ├── Load & Filter Data
   │   ├── Tokenize Texts
   │   └── Distribute to Clients (IID/Non-IID/By Author)
   ├── Initialize Server
   │   └── Create Global Model (GPT-2 + LoRA)
   └── Initialize Clients
       └── Assign Dataset to Each Client

2. TRAINING ROUNDS (Repeat N times)
   ├── Client Selection
   │   └── Randomly select K clients
   ├── Client Training
   │   ├── Download Global Model
   │   ├── Train Locally for E epochs
   │   ├── Track Metrics
   │   └── Upload LoRA Weights
   ├── Server Aggregation
   │   ├── Collect Weights from Selected Clients
   │   ├── FedAvg Aggregation (Weighted Average)
   │   └── Update Global Model
   └── Global Evaluation
       ├── Evaluate on Test Set
       └── Log Metrics (Loss, Perplexity)

3. FINALIZATION
   ├── Save Final Global Model
   ├── Save All Metrics (JSON)
   ├── Generate Plots (Loss, Perplexity curves)
   └── Test Text Generation
```

## 🚀 Quick Start

### Instalasi Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**

- transformers (Hugging Face)
- peft (LoRA implementation)
- datasets
- torch
- pandas, numpy
- matplotlib (visualization)

### Cara 1: Jalankan dengan Default Settings

```bash
python fed_learning/federated_learning.py
```

### Cara 2: Jalankan Example Script

```bash
python run_federated_example.py
```

### Cara 3: Custom Configuration

```python
from fed_learning import FederatedLearning, get_default_config

# Get config
config = get_default_config()

# Customize
config.federated.num_clients = 10
config.federated.num_rounds = 20
config.federated.data_distribution = "by_author"
config.training.num_epochs = 3

# Save config
config.save("my_config.json")

# Run
fed = FederatedLearning(config)
fed.setup()
fed.train()
```

## 🔧 Parameter Configuration

### Model Parameters

```python
config.model.model_name = "gpt2"  # atau "gpt2-medium", "gpt2-large"
config.model.max_length = 128     # sequence length
```

### LoRA Parameters

```python
config.lora.r = 8                           # rank (4, 8, 16, 32)
config.lora.lora_alpha = 16                 # scaling factor
config.lora.target_modules = ["c_attn", "c_proj"]  # layers untuk LoRA
config.lora.lora_dropout = 0.1              # dropout rate
```

### Training Parameters

```python
config.training.num_epochs = 3              # local epochs per round
config.training.batch_size = 8              # batch size
config.training.learning_rate = 5e-5        # learning rate
config.training.warmup_steps = 100          # warmup steps
config.training.fp16 = True                 # mixed precision (GPU only)
```

### Federated Parameters

```python
config.federated.num_clients = 5            # total clients
config.federated.num_rounds = 10            # federated rounds
config.federated.clients_per_round = 3      # clients selected per round
config.federated.data_distribution = "by_author"  # "iid", "non_iid", "by_author"
config.federated.aggregation_method = "fedavg"    # aggregation strategy
```

### Data Parameters

```python
config.data.dataset_path = "./twcs/twcs.csv"
config.data.text_column = "text"
config.data.author_column = "author_id"
config.data.filter_inbound_only = True      # only customer queries
config.data.filter_min_length = 10          # minimum text length
config.data.train_split = 0.8               # 80% training
```

## 📊 Output & Results

### Directory Structure

```
fed_results/
├── global_model/
│   ├── round_5/          # Checkpoint round 5
│   ├── round_10/         # Checkpoint round 10
│   └── round_final/      # Final model
├── client_0/
│   └── round_X/          # Client checkpoints
├── training_metrics.json  # Detailed metrics
├── global_metrics.json    # Global performance
└── training_progress.png  # Visualization plots

fed_logs/
└── client_X/             # Training logs per client
```

### Metrics JSON

```json
{
  "config": {...},
  "round_metrics": [
    {
      "round": 1,
      "selected_clients": [0, 2, 4],
      "global_metrics": {
        "eval_loss": 2.1234,
        "perplexity": 8.3456
      },
      "client_metrics": [...]
    }
  ],
  "training_history": {
    "rounds": [0, 1, 2, ...],
    "global_loss": [2.5, 2.3, 2.1, ...],
    "global_perplexity": [12.1, 9.9, 8.2, ...]
  }
}
```

## 📈 Monitoring & Visualization

### Console Output

```
====================================================================
FEDERATED LEARNING ROUND 3/10
====================================================================

[Round 3] Selected clients: [1, 3, 4]

----------------------------------------------------------------------
CLIENT TRAINING
----------------------------------------------------------------------
[Client 1] Starting training for 3 epochs...
[Client 1] Dataset size: 1250
[Client 1] Training complete! Loss: 2.0543

[Client 3] Starting training for 3 epochs...
[Client 3] Dataset size: 980
[Client 3] Training complete! Loss: 2.1234

----------------------------------------------------------------------
MODEL AGGREGATION
----------------------------------------------------------------------
Aggregating models using FedAvg...
Aggregated 24 parameters
Global model updated successfully!

----------------------------------------------------------------------
GLOBAL MODEL EVALUATION
----------------------------------------------------------------------
Global Evaluation Loss: 2.0123
Global Perplexity: 7.4821
====================================================================
```

### Visualization Plots

- **Loss Curve**: Global model loss over rounds
- **Perplexity Curve**: Model perplexity trend
- Automatically saved as PNG

## 🎯 Use Cases

### 1. Balanced IID Distribution

```python
config.federated.data_distribution = "iid"
config.federated.num_clients = 5
```

**Use case:** Testing baseline performance, equal data distribution

### 2. Heterogeneous Non-IID

```python
config.federated.data_distribution = "non_iid"
config.federated.num_clients = 5
```

**Use case:** Simulating different data characteristics per client

### 3. Real-world By Author

```python
config.federated.data_distribution = "by_author"
config.federated.num_clients = 10
config.federated.min_samples_per_client = 100
```

**Use case:** Real federated scenario, setiap client = company berbeda

## 🧪 Text Generation Testing

```python
# After training
fed_learning.test_generation([
    "Hello, I need help with",
    "My order is",
    "Can you tell me about",
])
```

**Output:**

```
====================================================================
TEXT GENERATION EXAMPLES
====================================================================

1. Prompt: Hello, I need help with
   Generated: Hello, I need help with my order. I placed an order...

2. Prompt: My order is
   Generated: My order is still pending. Can you check the status...
====================================================================
```

## 🔍 Advanced Features

### 1. Custom Aggregation Methods

Implementasi di `server.py`:

```python
def aggregate_custom(self, client_weights, client_sizes):
    # Your custom aggregation logic
    pass
```

### 2. Client Selection Strategies

Modify `server.py`:

```python
def select_clients_custom(self, num_clients, round_num):
    # Custom selection (e.g., based on performance)
    pass
```

### 3. Privacy-Preserving Techniques

Tambahkan differential privacy di `client.py`:

```python
# Add noise to gradients
weights = self.add_noise(weights, epsilon=0.1)
```

## 🐛 Troubleshooting

### Problem: Out of Memory

**Solution:**

```python
config.training.batch_size = 2  # Reduce batch size
config.model.max_length = 64    # Reduce sequence length
config.training.gradient_accumulation_steps = 4  # Use gradient accumulation
```

### Problem: Slow Training

**Solution:**

```python
config.training.fp16 = True     # Enable mixed precision
config.device = "cuda"          # Use GPU
config.training.num_epochs = 1  # Reduce local epochs
```

### Problem: Poor Convergence

**Solution:**

```python
config.training.learning_rate = 1e-4  # Increase LR
config.federated.num_rounds = 20      # More rounds
config.federated.clients_per_round = 5  # More clients per round
```

### Problem: Not Enough Authors

**Solution:**

```python
config.federated.min_samples_per_client = 50  # Lower threshold
config.federated.data_distribution = "non_iid"  # Use non-IID instead
```

## 📚 Technical Details

### LoRA (Low-Rank Adaptation)

- Freezes base GPT-2 weights
- Only trains small LoRA adapters (rank-decomposition matrices)
- Reduces trainable parameters by ~99%
- Faster training, lower memory usage

**Trainable Parameters:**

- GPT-2: ~124M total parameters
- LoRA (r=8): ~300K trainable parameters (~0.24%)

### FedAvg Aggregation

```
Global_weights = Σ (client_i_weights × n_i / n_total)

where:
- n_i = number of samples at client i
- n_total = total samples across selected clients
```

### Communication Efficiency

- Only LoRA weights transmitted (~300KB per round)
- Base model stays frozen (no communication)
- Significant bandwidth reduction vs. full model (~124MB)

## 🔐 Privacy Considerations

Current implementation:

- ✅ Only model weights shared (not raw data)
- ✅ Local training on client devices
- ✅ Minimal communication overhead

Potential additions:

- ⚠️ Differential Privacy (add noise to weights)
- ⚠️ Secure Aggregation (encrypted aggregation)
- ⚠️ Homomorphic Encryption

## 📝 Citation

Jika menggunakan sistem ini untuk penelitian:

```bibtex
@misc{federated_gpt2_lora,
  title={Federated Learning for GPT-2 Fine-tuning with LoRA},
  author={Your Name},
  year={2025},
  note={Twitter Customer Support Dataset}
}
```

## 📖 References

1. **LoRA**: [Hu et al., 2021 - LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
2. **Federated Learning**: [McMahan et al., 2017 - Communication-Efficient Learning of Deep Networks](https://arxiv.org/abs/1602.05629)
3. **GPT-2**: [Radford et al., 2019 - Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
4. **PEFT**: [Hugging Face PEFT Library](https://github.com/huggingface/peft)

## 🤝 Contributing

System ini dapat diperluas dengan:

- [ ] Support untuk model lain (GPT-Neo, BLOOM, LLaMA)
- [ ] Differential Privacy
- [ ] Secure Aggregation
- [ ] More aggregation methods (FedProx, FedNova)
- [ ] Client selection strategies
- [ ] Asynchronous federated learning

## 📞 Support

Untuk issues dan questions, silakan buat issue di repository.

---

**Happy Federated Learning! 🚀**

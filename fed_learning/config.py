"""
Configuration file for Federated Learning with GPT-2 and LoRA
"""
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration"""
    r: int = 8  # Rank of the update matrices
    lora_alpha: int = 16  # Scaling factor
    target_modules: List[str] = field(default_factory=lambda: ["c_attn", "c_proj"])
    lora_dropout: float = 0.1
    bias: str = "none"  # Options: "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "gpt2"  # Base model
    max_length: int = 128  # Maximum sequence length
    cache_dir: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration for each client"""
    num_epochs: int = 3  # Local epochs per round
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    fp16: bool = False  # Mixed precision training
    # Shuffle and subsample to speed up local training
    shuffle_dataset: bool = True
    max_samples_per_client: Optional[int] = None  # If set, limit samples per client for faster runs


@dataclass
class FederatedConfig:
    """Federated Learning configuration"""
    num_clients: int = 10  # Number of clients
    num_rounds: int = 5  # Number of federated rounds
    clients_per_round: int = 3  # Clients selected per round
    data_distribution: str = "iid"  # Options: "iid", "non_iid", "by_author"
    aggregation_method: str = "fedavg"  # Options: "fedavg", "fedprox"
    min_samples_per_client: int = 100


@dataclass
class DataConfig:
    """Data configuration"""
    dataset_path: str = "./twcs/twcs.csv"
    text_column: str = "text"
    author_column: str = "author_id"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    filter_min_length: int = 10  # Minimum text length
    filter_inbound_only: bool = True  # Only use customer queries


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    experiment_name: str = "federated_gpt2_lora"
    output_dir: str = "./fed_results"
    log_dir: str = "./fed_logs"
    seed: int = 42
    device: str = "cuda"  # Options: "cuda", "cpu", "mps"
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            'experiment_name': self.experiment_name,
            'output_dir': self.output_dir,
            'log_dir': self.log_dir,
            'seed': self.seed,
            'device': self.device,
            'model': self.model.__dict__,
            'lora': self.lora.__dict__,
            'training': self.training.__dict__,
            'federated': self.federated.__dict__,
            'data': self.data.__dict__,
        }
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        config = cls(
            experiment_name=config_dict.get('experiment_name', 'federated_gpt2_lora'),
            output_dir=config_dict.get('output_dir', './fed_results'),
            log_dir=config_dict.get('log_dir', './fed_logs'),
            seed=config_dict.get('seed', 42),
            device=config_dict.get('device', 'cuda'),
        )
        
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'lora' in config_dict:
            config.lora = LoRAConfig(**config_dict['lora'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'federated' in config_dict:
            config.federated = FederatedConfig(**config_dict['federated'])
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        
        print(f"Configuration loaded from {path}")
        return config
    
    def __str__(self):
        """String representation of configuration"""
        return f"""
Experiment Configuration: {self.experiment_name}
{'='*60}
Output Directory: {self.output_dir}
Log Directory: {self.log_dir}
Device: {self.device}
Seed: {self.seed}

Model Config:
  - Model Name: {self.model.model_name}
  - Max Length: {self.model.max_length}

LoRA Config:
  - Rank (r): {self.lora.r}
  - Alpha: {self.lora.lora_alpha}
  - Target Modules: {self.lora.target_modules}
  - Dropout: {self.lora.lora_dropout}

Training Config:
  - Epochs per Round: {self.training.num_epochs}
  - Batch Size: {self.training.batch_size}
  - Learning Rate: {self.training.learning_rate}
  - Weight Decay: {self.training.weight_decay}

Federated Config:
  - Number of Clients: {self.federated.num_clients}
  - Number of Rounds: {self.federated.num_rounds}
  - Clients per Round: {self.federated.clients_per_round}
  - Data Distribution: {self.federated.data_distribution}
  - Aggregation Method: {self.federated.aggregation_method}

Data Config:
  - Dataset Path: {self.data.dataset_path}
  - Text Column: {self.data.text_column}
  - Train/Val/Test Split: {self.data.train_split}/{self.data.val_split}/{self.data.test_split}
  - Filter Inbound Only: {self.data.filter_inbound_only}
{'='*60}
"""


# Default configuration instance
def get_default_config() -> ExperimentConfig:
    """Get default configuration"""
    return ExperimentConfig()


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    print(config)
    
    # Save configuration
    config.save("fed_config.json")
    
    # Load configuration
    loaded_config = ExperimentConfig.load("fed_config.json")
    print("\nLoaded configuration:")
    print(loaded_config)

"""
Server implementation for Federated Learning
Manages global model and coordinates client training
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import os
import json
import numpy as np
from typing import Dict, List, Optional
from collections import OrderedDict
import copy


class FederatedServer:
    """Server for federated learning coordination and model aggregation"""
    
    def __init__(self, config):
        """
        Initialize federated server
        
        Args:
            config: ExperimentConfig object
        """
        self.config = config
        self.global_model = None
        self.tokenizer = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.current_round = 0
        self.global_metrics = []
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        
    def initialize_global_model(self):
        """Initialize the global model with LoRA adapters"""
        print("\n" + "="*60)
        print("Initializing Global Model")
        print("="*60)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir,
            torch_dtype=torch.float16 if self.config.training.fp16 else torch.float32,
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.lora_alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.lora_dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA
        self.global_model = get_peft_model(base_model, peft_config)
        self.global_model.to(self.device)
        
        print(f"Model loaded: {self.config.model.model_name}")
        self.global_model.print_trainable_parameters()
        print("="*60)
        
    def get_global_model(self):
        """Get a copy of the global model"""
        if self.global_model is None:
            raise ValueError("Global model not initialized.")
        return copy.deepcopy(self.global_model)
    
    def select_clients(self, num_clients: int, round_num: int) -> List[int]:
        """
        Select clients for training in this round
        
        Args:
            num_clients: Total number of available clients
            round_num: Current round number
            
        Returns:
            List of selected client IDs
        """
        np.random.seed(self.config.seed + round_num)
        
        num_selected = min(self.config.federated.clients_per_round, num_clients)
        selected = np.random.choice(num_clients, num_selected, replace=False)
        
        print(f"\n[Round {round_num}] Selected clients: {selected.tolist()}")
        return selected.tolist()
    
    def aggregate_fedavg(self, client_weights: List[Dict[str, torch.Tensor]], 
                        client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using FedAvg (weighted average)
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated weights
        """
        print("\nAggregating models using FedAvg...")
        
        # Calculate weights based on dataset sizes
        total_size = sum(client_sizes)
        weight_factors = [size / total_size for size in client_sizes]
        
        # Initialize aggregated weights
        aggregated_weights = OrderedDict()
        
        # Get all parameter names from first client
        param_names = list(client_weights[0].keys())
        
        # Aggregate each parameter
        for param_name in param_names:
            # Weighted sum of parameters
            aggregated_param = sum(
                client_weights[i][param_name] * weight_factors[i]
                for i in range(len(client_weights))
            )
            aggregated_weights[param_name] = aggregated_param
        
        print(f"Aggregated {len(aggregated_weights)} parameters")
        return aggregated_weights
    
    def aggregate_models(self, client_weights: List[Dict[str, torch.Tensor]], 
                        client_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models based on configured method
        
        Args:
            client_weights: List of client model weights
            client_sizes: List of client dataset sizes
            
        Returns:
            Aggregated weights
        """
        method = self.config.federated.aggregation_method.lower()
        
        if method == "fedavg":
            return self.aggregate_fedavg(client_weights, client_sizes)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated weights
        
        Args:
            aggregated_weights: Aggregated weights from clients
        """
        if self.global_model is None:
            raise ValueError("Global model not initialized.")
        
        print("Updating global model...")
        
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if param.requires_grad and name in aggregated_weights:
                    param.copy_(aggregated_weights[name].to(self.device))
        
        print("Global model updated successfully!")
    
    def evaluate_global_model(self, test_dataset) -> Dict:
        """
        Evaluate the global model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        
        print("\n" + "="*60)
        print(f"Evaluating Global Model - Round {self.current_round}")
        print("="*60)
        
        if self.global_model is None:
            raise ValueError("Global model not initialized.")
        
        # Create temporary trainer for evaluation
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config.output_dir, "temp"),
            per_device_eval_batch_size=self.config.training.batch_size,
            fp16=self.config.training.fp16 and torch.cuda.is_available(),
            report_to="none",
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=self.global_model,
            args=training_args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        
        eval_result = trainer.evaluate()
        
        metrics = {
            'round': self.current_round,
            'eval_loss': eval_result['eval_loss'],
            'perplexity': torch.exp(torch.tensor(eval_result['eval_loss'])).item(),
        }
        
        self.global_metrics.append(metrics)
        
        print(f"Global Evaluation Loss: {metrics['eval_loss']:.4f}")
        print(f"Global Perplexity: {metrics['perplexity']:.4f}")
        print("="*60)
        
        return metrics
    
    def save_global_model(self, round_num: Optional[int] = None):
        """
        Save the global model
        
        Args:
            round_num: Round number (if None, uses current_round)
        """
        if self.global_model is None:
            raise ValueError("Global model not initialized.")
        
        round_num = round_num or self.current_round
        
        save_dir = os.path.join(
            self.config.output_dir,
            "global_model",
            f"round_{round_num}"
        )
        os.makedirs(save_dir, exist_ok=True)
        
        self.global_model.save_pretrained(save_dir)
        print(f"\nGlobal model saved to {save_dir}")
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics_path = os.path.join(self.config.output_dir, "global_metrics.json")
        
        with open(metrics_path, 'w') as f:
            json.dump(self.global_metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_path}")
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the global model
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        if self.global_model is None:
            raise ValueError("Global model not initialized.")
        
        self.global_model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.global_model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*60)
        print("FEDERATED LEARNING SUMMARY")
        print("="*60)
        print(f"Total Rounds: {self.current_round}")
        print(f"Configuration: {self.config.experiment_name}")
        print(f"\nGlobal Model Metrics:")
        
        if self.global_metrics:
            for metric in self.global_metrics:
                print(f"  Round {metric['round']}: "
                      f"Loss={metric['eval_loss']:.4f}, "
                      f"Perplexity={metric['perplexity']:.4f}")
        
        print("="*60)
    
    def increment_round(self):
        """Increment the current round counter"""
        self.current_round += 1


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    from dataset import FederatedDataset
    
    config = get_default_config()
    
    # Initialize server
    server = FederatedServer(config)
    server.initialize_global_model()
    
    # Prepare dataset
    fed_dataset = FederatedDataset(config)
    client_datasets, test_dataset = fed_dataset.prepare_datasets()
    
    # Evaluate initial model
    metrics = server.evaluate_global_model(test_dataset)
    print(f"\nInitial metrics: {metrics}")
    
    # Generate sample text
    prompt = "Hello, I need help with"
    generated = server.generate_text(prompt)
    print(f"\nGenerated text:\n{generated}")
    
    # Save model
    server.save_global_model(round_num=0)

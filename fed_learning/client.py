"""
Client implementation for Federated Learning
Each client trains a local model on its own data
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import os
from typing import Dict, Optional
import copy


class FederatedClient:
    """Client for federated learning with LoRA fine-tuning"""
    
    def __init__(self, client_id: int, config, dataset=None):
        """
        Initialize federated client
        
        Args:
            client_id: Unique identifier for this client
            config: ExperimentConfig object
            dataset: Training dataset for this client
        """
        self.client_id = client_id
        self.config = config
        self.dataset = dataset
        self.model = None
        self.tokenizer = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Training metrics
        self.training_history = []
        self.current_round = 0
        
    def setup_model(self, global_model=None):
        """
        Setup model with LoRA adapters
        
        Args:
            global_model: Global model to initialize from (if None, loads from scratch)
        """
        print(f"\n[Client {self.client_id}] Setting up model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            cache_dir=self.config.model.cache_dir
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if global_model is None:
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
            self.model = get_peft_model(base_model, peft_config)
        else:
            # Use global model (already has LoRA adapters)
            self.model = copy.deepcopy(global_model)
        
        self.model.to(self.device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Train the local model
        
        Args:
            num_epochs: Number of epochs (if None, uses config)
            
        Returns:
            Training metrics dictionary
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        
        if self.dataset is None:
            raise ValueError("No dataset provided for training.")
        
        num_epochs = num_epochs or self.config.training.num_epochs
        
        print(f"\n[Client {self.client_id}] Starting training for {num_epochs} epochs...")

        # Prepare dataset: shuffle and optionally subsample to speed up training
        dataset = self.dataset
        try:
            if getattr(self.config.training, "shuffle_dataset", True):
                dataset = dataset.shuffle(seed=self.config.seed)
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: failed to shuffle dataset: {e}")

        max_samples = getattr(self.config.training, "max_samples_per_client", None)
        if max_samples is not None and len(dataset) > max_samples:
            try:
                dataset = dataset.select(list(range(int(max_samples))))
                print(f"[Client {self.client_id}] Subsampled dataset to {len(dataset)} samples for faster training")
            except Exception as e:
                print(f"[Client {self.client_id}] Warning: failed to subsample dataset: {e}")

        print(f"[Client {self.client_id}] Dataset size: {len(dataset)}")
        
        # Create output directory for this client
        output_dir = os.path.join(
            self.config.output_dir,
            f"client_{self.client_id}",
            f"round_{self.current_round}"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_steps=self.config.training.warmup_steps,
            max_grad_norm=self.config.training.max_grad_norm,
            logging_dir=os.path.join(self.config.log_dir, f"client_{self.client_id}"),
            logging_steps=self.config.training.logging_steps,
            save_steps=self.config.training.save_steps,
            save_total_limit=2,
            fp16=self.config.training.fp16 and torch.cuda.is_available(),
            report_to="none",  # Disable wandb, tensorboard, etc.
            remove_unused_columns=False,
        )
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        
        # Train
        train_result = trainer.train()
        
        # Get metrics
        metrics = {
            'client_id': self.client_id,
            'round': self.current_round,
            'train_loss': train_result.training_loss,
            'train_samples': len(self.dataset),
        }
        
        self.training_history.append(metrics)
        
        print(f"[Client {self.client_id}] Training complete!")
        print(f"[Client {self.client_id}] Loss: {metrics['train_loss']:.4f}")
        
        return metrics
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """
        Get LoRA adapter weights from the model
        
        Returns:
            Dictionary of trainable parameter names and weights
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                weights[name] = param.data.cpu().clone()
        
        return weights
    
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """
        Set LoRA adapter weights in the model
        
        Args:
            weights: Dictionary of parameter names and weights
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in weights:
                    param.copy_(weights[name].to(self.device))
    
    def evaluate(self, test_dataset) -> Dict:
        """
        Evaluate the model on test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        print(f"\n[Client {self.client_id}] Evaluating model...")
        
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
            model=self.model,
            args=training_args,
            eval_dataset=test_dataset,
            data_collator=data_collator,
        )
        
        eval_result = trainer.evaluate()
        
        metrics = {
            'client_id': self.client_id,
            'eval_loss': eval_result['eval_loss'],
            'perplexity': torch.exp(torch.tensor(eval_result['eval_loss'])).item(),
        }
        
        print(f"[Client {self.client_id}] Evaluation Loss: {metrics['eval_loss']:.4f}")
        print(f"[Client {self.client_id}] Perplexity: {metrics['perplexity']:.4f}")
        
        return metrics
    
    def generate_text(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using the local model
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        self.model.eval()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def save_model(self, save_dir: str):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        save_path = os.path.join(save_dir, f"client_{self.client_id}")
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        print(f"[Client {self.client_id}] Model saved to {save_path}")
    
    def increment_round(self):
        """Increment the current round counter"""
        self.current_round += 1


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    from dataset import FederatedDataset
    
    config = get_default_config()
    config.training.num_epochs = 1
    
    # Prepare dataset
    fed_dataset = FederatedDataset(config)
    client_datasets, test_dataset = fed_dataset.prepare_datasets()
    
    # Create and train a client
    client = FederatedClient(client_id=0, config=config, dataset=client_datasets[0])
    client.setup_model()
    
    # Train
    metrics = client.train()
    print(f"\nTraining metrics: {metrics}")
    
    # Evaluate
    eval_metrics = client.evaluate(test_dataset)
    print(f"\nEvaluation metrics: {eval_metrics}")
    
    # Generate text
    sample_prompt = "Hello, I need help with"
    generated = client.generate_text(sample_prompt)
    print(f"\nGenerated text:\n{generated}")

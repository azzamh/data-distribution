"""
Federated Learning Main Orchestrator
Coordinates the entire federated learning process
"""
import torch
import numpy as np
import random
import os
import json
from datetime import datetime
from typing import List, Dict
import matplotlib.pyplot as plt

from config import ExperimentConfig, get_default_config
from dataset import FederatedDataset
from client import FederatedClient
from server import FederatedServer


class FederatedLearning:
    """Main class for orchestrating federated learning"""
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize federated learning system
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.server = None
        self.clients = []
        self.dataset_manager = None
        self.test_dataset = None
        
        # Set random seeds for reproducibility
        self.set_seed(config.seed)
        
        # Training history
        self.training_history = {
            'rounds': [],
            'global_loss': [],
            'global_perplexity': [],
            'client_metrics': []
        }
        
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")
    
    def setup(self):
        """Setup federated learning system"""
        print("\n" + "="*70)
        print("FEDERATED LEARNING SETUP")
        print("="*70)
        print(self.config)
        
        # Initialize dataset manager
        print("\nStep 1: Preparing Datasets")
        print("-"*70)
        self.dataset_manager = FederatedDataset(self.config)
        client_datasets, self.test_dataset = self.dataset_manager.prepare_datasets()
        
        # Initialize server
        print("\nStep 2: Initializing Server")
        print("-"*70)
        self.server = FederatedServer(self.config)
        self.server.initialize_global_model()
        
        # Initialize clients
        print("\nStep 3: Initializing Clients")
        print("-"*70)
        self.clients = []
        for client_id in range(self.config.federated.num_clients):
            client = FederatedClient(
                client_id=client_id,
                config=self.config,
                dataset=client_datasets[client_id]
            )
            self.clients.append(client)
            print(f"Client {client_id} initialized with {len(client_datasets[client_id])} samples")
            # print(f" - Dataset samples: {(client_datasets[client_id][0])}")
        
        print("\n" + "="*70)
        print("SETUP COMPLETE!")
        print("="*70)
    
    def run_round(self, round_num: int) -> Dict:
        """
        Execute one federated learning round
        
        Args:
            round_num: Current round number
            
        Returns:
            Round metrics
        """
        print("\n" + "="*70)
        print(f"FEDERATED LEARNING ROUND {round_num + 1}/{self.config.federated.num_rounds}")
        print("="*70)
        
        # Select clients for this round
        selected_client_ids = self.server.select_clients(
            num_clients=len(self.clients),
            round_num=round_num
        )
        
        # Store client weights and dataset sizes
        client_weights = []
        client_sizes = []
        client_round_metrics = []
        
        # Train selected clients
        print("\n" + "-"*70)
        print("CLIENT TRAINING")
        print("-"*70)
        
        for client_id in selected_client_ids:
            client = self.clients[client_id]
            
            # Setup model with current global weights
            client.setup_model(global_model=self.server.get_global_model())
            client.current_round = round_num
            
            # Train
            metrics = client.train()
            client_round_metrics.append(metrics)
            
            # Get weights
            weights = client.get_model_weights()
            client_weights.append(weights)
            client_sizes.append(len(client.dataset))
        
        # Aggregate models
        print("\n" + "-"*70)
        print("MODEL AGGREGATION")
        print("-"*70)
        
        aggregated_weights = self.server.aggregate_models(
            client_weights=client_weights,
            client_sizes=client_sizes
        )
        
        # Update global model
        self.server.update_global_model(aggregated_weights)
        self.server.current_round = round_num + 1
        
        # Evaluate global model
        print("\n" + "-"*70)
        print("GLOBAL MODEL EVALUATION")
        print("-"*70)
        
        global_metrics = self.server.evaluate_global_model(self.test_dataset)
        
        # Compile round metrics
        round_metrics = {
            'round': round_num + 1,
            'selected_clients': selected_client_ids,
            'global_metrics': global_metrics,
            'client_metrics': client_round_metrics,
            'avg_client_loss': np.mean([m['train_loss'] for m in client_round_metrics]),
        }
        
        # Update history
        self.training_history['rounds'].append(round_num + 1)
        self.training_history['global_loss'].append(global_metrics['eval_loss'])
        self.training_history['global_perplexity'].append(global_metrics['perplexity'])
        self.training_history['client_metrics'].append(client_round_metrics)
        
        # Print round summary
        print("\n" + "-"*70)
        print("ROUND SUMMARY")
        print("-"*70)
        print(f"Global Loss: {global_metrics['eval_loss']:.4f}")
        print(f"Global Perplexity: {global_metrics['perplexity']:.4f}")
        print(f"Average Client Loss: {round_metrics['avg_client_loss']:.4f}")
        print("="*70)
        
        return round_metrics
    
    def train(self):
        """Execute full federated learning training"""
        print("\n" + "="*70)
        print("STARTING FEDERATED LEARNING TRAINING")
        print("="*70)
        
        start_time = datetime.now()
        
        # Evaluate initial global model
        print("\n" + "="*70)
        print("INITIAL GLOBAL MODEL EVALUATION")
        print("="*70)
        initial_metrics = self.server.evaluate_global_model(self.test_dataset)
        self.training_history['rounds'].append(0)
        self.training_history['global_loss'].append(initial_metrics['eval_loss'])
        self.training_history['global_perplexity'].append(initial_metrics['perplexity'])
        
        # Run federated rounds
        all_round_metrics = []
        
        for round_num in range(self.config.federated.num_rounds):
            round_metrics = self.run_round(round_num)
            all_round_metrics.append(round_metrics)
            
            # Save global model periodically
            if (round_num + 1) % 5 == 0:
                self.server.save_global_model(round_num + 1)
        
        # Save final model
        print("\n" + "="*70)
        print("SAVING FINAL MODEL")
        print("="*70)
        self.server.save_global_model()
        
        # Save metrics
        self.save_results(all_round_metrics)
        
        # Print summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("FEDERATED LEARNING COMPLETED!")
        print("="*70)
        print(f"Total Duration: {duration}")
        print(f"Total Rounds: {self.config.federated.num_rounds}")
        print(f"Initial Loss: {self.training_history['global_loss'][0]:.4f}")
        print(f"Final Loss: {self.training_history['global_loss'][-1]:.4f}")
        print(f"Initial Perplexity: {self.training_history['global_perplexity'][0]:.4f}")
        print(f"Final Perplexity: {self.training_history['global_perplexity'][-1]:.4f}")
        print("="*70)
        
        # Plot results
        self.plot_results()
        
        return all_round_metrics
    
    def save_results(self, round_metrics: List[Dict]):
        """Save training results"""
        results_dir = self.config.output_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed metrics
        metrics_path = os.path.join(results_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'config': {
                    'experiment_name': self.config.experiment_name,
                    'num_rounds': self.config.federated.num_rounds,
                    'num_clients': self.config.federated.num_clients,
                    'data_distribution': self.config.federated.data_distribution,
                },
                'round_metrics': round_metrics,
                'training_history': self.training_history,
            }, f, indent=2)
        
        print(f"\nResults saved to {metrics_path}")
        
        # Save server metrics
        self.server.save_metrics()
    
    def plot_results(self):
        """Plot training results"""
        results_dir = self.config.output_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        axes[0].plot(self.training_history['rounds'], 
                    self.training_history['global_loss'], 
                    marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Round', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Global Model Loss over Rounds', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Plot perplexity
        axes[1].plot(self.training_history['rounds'], 
                    self.training_history['global_perplexity'], 
                    marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Round', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title('Global Model Perplexity over Rounds', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Training plot saved to {plot_path}")
        
        plt.close()
    
    def test_generation(self, prompts: List[str] = None):
        """
        Test text generation with the final model
        
        Args:
            prompts: List of prompts to test (uses defaults if None)
        """
        if prompts is None:
            prompts = [
                "Hello, I need help with",
                "My order is",
                "Can you tell me about",
                "I have a problem with",
            ]
        
        print("\n" + "="*70)
        print("TEXT GENERATION EXAMPLES")
        print("="*70)
        
        for i, prompt in enumerate(prompts, 1):
            generated = self.server.generate_text(prompt, max_length=50)
            print(f"\n{i}. Prompt: {prompt}")
            print(f"   Generated: {generated}")
        
        print("="*70)


def main():
    """Main function to run federated learning"""
    # Load or create configuration
    config_path = "fed_config.json"
    
    if os.path.exists(config_path):
        print(f"Loading configuration from {config_path}")
        config = ExperimentConfig.load(config_path)
    else:
        print("Creating default configuration")
        config = get_default_config()
        
        # Customize configuration
        config.federated.num_rounds = 5
        config.federated.num_clients = 5
        config.federated.clients_per_round = 3
        config.federated.data_distribution = "by_author"  # Options: "iid", "non_iid", "by_author"
        config.training.num_epochs = 2
        config.training.batch_size = 4
        
        # Save configuration
        config.save(config_path)
    
    # Create federated learning system
    fed_learning = FederatedLearning(config)
    
    # Setup
    fed_learning.setup()
    
    # Train
    # fed_learning.train()
    
    # Test generation
    # fed_learning.test_generation()
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

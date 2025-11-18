"""
Example script untuk menjalankan Federated Learning GPT-2 dengan LoRA
Simplified version untuk testing cepat
"""

from fed_learning import FederatedLearning, get_default_config


def run_federated_learning_example():
    """Run a simple federated learning example"""
    
    print("="*70)
    print("FEDERATED LEARNING GPT-2 + LoRA - EXAMPLE")
    print("="*70)
    
    # Get default configuration
    config = get_default_config()
    
    # Customize for quick testing
    config.experiment_name = "fed_gpt2_example"
    config.output_dir = "./fed_results_example"
    config.log_dir = "./fed_logs_example"
    
    # Model config
    config.model.model_name = "gpt2"  # Use base GPT-2
    config.model.max_length = 128
    
    # LoRA config
    config.lora.r = 8
    config.lora.lora_alpha = 16
    config.lora.target_modules = ["c_attn", "c_proj"]
    
    # Training config - reduced for quick testing
    config.training.num_epochs = 1  # Quick local training
    config.training.batch_size = 4
    config.training.learning_rate = 5e-5
    
    # Federated config
    config.federated.num_clients = 5
    config.federated.num_rounds = 3  # Short example
    config.federated.clients_per_round = 3
    config.federated.data_distribution = "by_author"  # Natural non-IID
    
    # Data config
    config.data.dataset_path = "./twcs/twcs.csv"
    config.data.filter_inbound_only = True  # Only customer queries
    config.data.filter_min_length = 10
    
    # Print configuration
    print("\nConfiguration:")
    print(config)
    
    # Save configuration
    config.save("fed_config_example.json")
    
    # Create federated learning system
    print("\nInitializing Federated Learning System...")
    fed_learning = FederatedLearning(config)
    
    # Setup
    print("\nSetting up system...")
    fed_learning.setup()
    
    # Train
    print("\nStarting training...")
    fed_learning.train()
    
    # Test generation
    print("\nTesting text generation...")
    test_prompts = [
        "Hello, I need help with",
        "My order is",
        "Can you tell me",
        "I have a problem",
    ]
    fed_learning.test_generation(test_prompts)
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {config.output_dir}")
    print(f"Configuration saved to: fed_config_example.json")


if __name__ == "__main__":
    run_federated_learning_example()

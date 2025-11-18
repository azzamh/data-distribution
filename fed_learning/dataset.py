"""
Dataset utilities for Federated Learning
Handles data loading, preprocessing, and distribution across clients
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from transformers import AutoTokenizer
import torch
from collections import defaultdict


class FederatedDataset:
    """Manages dataset distribution for federated learning"""
    
    def __init__(self, config):
        """
        Initialize federated dataset
        
        Args:
            config: ExperimentConfig object containing all configurations
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            cache_dir=config.model.cache_dir
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = None
        self.client_datasets = {}
        self.test_dataset = None
        
    def load_data(self) -> pd.DataFrame:
        """Load data from CSV file"""
        print(f"Loading data from {self.config.data.dataset_path}")
        df = pd.read_csv(self.config.data.dataset_path)
        
        # Filter data
        if self.config.data.filter_inbound_only and 'inbound' in df.columns:
            df = df[df['inbound'] == True]
            print(f"Filtered to inbound messages only: {len(df)} samples")
        
        # Filter by text length
        if self.config.data.text_column in df.columns:
            df = df[df[self.config.data.text_column].str.len() >= self.config.data.filter_min_length]
            print(f"Filtered by minimum length: {len(df)} samples")
        
        # Remove missing values
        df = df.dropna(subset=[self.config.data.text_column])
        
        print(f"Total samples after filtering: {len(df)}")
        return df
    
    def preprocess_function(self, examples):
        """Tokenize and prepare data"""
        # Tokenize the texts
        result = self.tokenizer(
            examples[self.config.data.text_column],
            truncation=True,
            max_length=self.config.model.max_length,
            padding='max_length',
        )
        
        # For causal language modeling, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        
        return result
    
    def split_iid(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Split data in IID (Independent and Identically Distributed) manner
        Each client gets random samples
        """
        print("\nSplitting data in IID manner...")
        
        # Shuffle data
        df = df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        
        # Calculate samples per client
        total_samples = len(df)
        samples_per_client = total_samples // self.config.federated.num_clients
        
        client_data = {}
        for i in range(self.config.federated.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.config.federated.num_clients - 1 else total_samples
            
            client_data[i] = df.iloc[start_idx:end_idx].reset_index(drop=True)
            print(f"Client {i}: {len(client_data[i])} samples")
        
        return client_data
    
    def split_non_iid(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Split data in Non-IID manner
        Create data heterogeneity by sorting and chunking
        """
        print("\nSplitting data in Non-IID manner...")
        
        # Sort by text length to create heterogeneity
        df['text_length'] = df[self.config.data.text_column].str.len()
        df = df.sort_values('text_length').reset_index(drop=True)
        
        # Create chunks
        chunk_size = len(df) // (self.config.federated.num_clients * 2)
        chunks = []
        
        for i in range(0, len(df), chunk_size):
            chunks.append(df.iloc[i:i+chunk_size])
        
        # Assign chunks to clients (each client gets 2 random chunks)
        np.random.seed(self.config.seed)
        chunk_indices = np.random.permutation(len(chunks))
        
        client_data = {}
        for i in range(self.config.federated.num_clients):
            # Get 2 chunks per client
            client_chunks_idx = chunk_indices[i*2:(i+1)*2]
            client_df = pd.concat([chunks[idx] for idx in client_chunks_idx if idx < len(chunks)])
            client_data[i] = client_df.reset_index(drop=True)
            print(f"Client {i}: {len(client_data[i])} samples")
        
        return client_data
    
    def split_by_author(self, df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Split data by author (natural non-IID distribution)
        Each client represents different authors/companies
        """
        print("\nSplitting data by author...")
        
        if self.config.data.author_column not in df.columns:
            print(f"Warning: {self.config.data.author_column} not found. Falling back to IID split.")
            return self.split_iid(df)
        
        # Get top authors
        author_counts = df[self.config.data.author_column].value_counts()
        print(f"\nTotal unique authors: {len(author_counts)}")
        
        # Filter authors with minimum samples
        valid_authors = author_counts[
            author_counts >= self.config.federated.min_samples_per_client
        ].index.tolist()
        
        print(f"Authors with >= {self.config.federated.min_samples_per_client} samples: {len(valid_authors)}")
        
        if len(valid_authors) < self.config.federated.num_clients:
            print(f"Warning: Not enough authors. Using top {self.config.federated.num_clients} authors.")
            valid_authors = author_counts.head(self.config.federated.num_clients).index.tolist()
        
        # Select top N authors for N clients
        selected_authors = valid_authors[:self.config.federated.num_clients]
        
        client_data = {}
        for i, author in enumerate(selected_authors):
            client_df = df[df[self.config.data.author_column] == author].reset_index(drop=True)
            client_data[i] = client_df
            print(f"Client {i} (Author: {author}): {len(client_df)} samples")
        
        return client_data
    
    # Add new helper to summarize author distribution across clients
    def print_author_stats(self, client_data: Dict[int, pd.DataFrame], top_n: int = 5):
        """
        Print author statistics for each client and overall top authors across all clients.
        This uses the raw pandas DataFrames in client_data (must include author column).
        """
        author_col = self.config.data.author_column
        if author_col not in next(iter(client_data.values())).columns:
            print(f"Author column '{author_col}' is not present in client data. Skipping author stats.")
            return

        print("\nAuthor distribution per client (top authors):")
        total_authors_all_clients = defaultdict(int)
        for client_id, cdf in client_data.items():
            total = len(cdf)
            if total == 0:
                print(f"Client {client_id}: 0 samples")
                continue

            counts = cdf[author_col].value_counts()
            unique = len(counts)
            top = counts.head(top_n)
            # collect global counts
            for author, cnt in counts.items():
                total_authors_all_clients[author] += cnt

            top_str = ", ".join(
                [f"{a}: {cnt} ({cnt/total*100:.1f}%)" for a, cnt in top.items()]
            )
            print(f"Client {client_id}: total={total} | unique_authors={unique} | top{top_n}=[{top_str}]")

        # Print overall top authors across all clients
        print("\nOverall top authors across all clients:")
        if total_authors_all_clients:
            global_counts = pd.Series(total_authors_all_clients).sort_values(ascending=False)
            global_top = global_counts.head(top_n)
            global_total = global_counts.sum()
            for author, cnt in global_top.items():
                print(f"{author}: {cnt} ({cnt/global_total*100:.1f}%)")
        else:
            print("No authors found across clients.")

    def prepare_datasets(self) -> Tuple[Dict[int, Dataset], Dataset]:
        """
        Prepare datasets for all clients and test set
        
        Returns:
            Tuple of (client_datasets, test_dataset)
        """
        # Load data
        df = self.load_data()
        
        # Split train/test
        train_size = int(len(df) * self.config.data.train_split)
        test_size = int(len(df) * self.config.data.test_split)
        
        df = df.sample(frac=1, random_state=self.config.seed).reset_index(drop=True)
        
        train_df = df.iloc[:train_size]
        test_df = df.iloc[-test_size:]
        
        print(f"\nTrain samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
        
        # Split training data across clients based on distribution strategy
        distribution_method = self.config.federated.data_distribution.lower()
        
        if distribution_method == "iid":
            client_data = self.split_iid(train_df)
        elif distribution_method == "non_iid":
            client_data = self.split_non_iid(train_df)
        elif distribution_method == "by_author":
            client_data = self.split_by_author(train_df)
        else:
            raise ValueError(f"Unknown distribution method: {distribution_method}")
        
        # Print author distribution stats for debugging/analysis
        try:
            self.print_author_stats(client_data)
        except Exception as e:
            print(f"Could not print author stats: {e}")
        
        # Convert to HuggingFace datasets and tokenize
        print("\nTokenizing datasets...")
        client_datasets = {}
        
        for client_id, client_df in client_data.items():
            dataset = Dataset.from_pandas(client_df[[self.config.data.text_column]])
            tokenized_dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Tokenizing client {client_id}"
            )
            client_datasets[client_id] = tokenized_dataset
        
        # Prepare test dataset
        test_dataset = Dataset.from_pandas(test_df[[self.config.data.text_column]])
        test_dataset = test_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=test_dataset.column_names,
            desc="Tokenizing test set"
        )
        
        self.client_datasets = client_datasets
        self.test_dataset = test_dataset
        
        print("\n" + "="*60)
        print("Dataset Preparation Complete!")
        print("="*60)
        for client_id, dataset in client_datasets.items():
            print(f"Client {client_id}: {len(dataset)} samples")
        print(f"Test set: {len(test_dataset)} samples")
        print("="*60)
        
        return client_datasets, test_dataset
    
    def get_client_dataset(self, client_id: int) -> Dataset:
        """Get dataset for specific client"""
        if client_id not in self.client_datasets:
            raise ValueError(f"Client {client_id} not found")
        return self.client_datasets[client_id]
    
    def get_test_dataset(self) -> Dataset:
        """Get test dataset"""
        return self.test_dataset


if __name__ == "__main__":
    # Example usage
    from config import get_default_config
    
    config = get_default_config()
    fed_dataset = FederatedDataset(config)
    client_datasets, test_dataset = fed_dataset.prepare_datasets()
    
    # Show sample
    print("\nSample from Client 0:")
    sample = client_datasets[0][0]
    print(f"Input IDs shape: {len(sample['input_ids'])}")
    print(f"Decoded text: {fed_dataset.tokenizer.decode(sample['input_ids'][:50])}")

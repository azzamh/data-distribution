import pandas as pd
from datasets import Dataset, DatasetDict

# Load the CLINC150 dataset using pandas from Hugging Face
# The dataset has been converted to parquet format
# Note: CLINC150 has multiple configurations, we'll try the default one
train_url = "https://huggingface.co/datasets/DeepPavlov/clinc150/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
validation_url = "https://huggingface.co/datasets/DeepPavlov/clinc150/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
test_url = "https://huggingface.co/datasets/DeepPavlov/clinc150/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"

print("Loading train data...")
train_df = pd.read_parquet(train_url)
print("Loading validation data...")
validation_df = pd.read_parquet(validation_url)
print("Loading test data...")
test_df = pd.read_parquet(test_url)

# Convert to Hugging Face Dataset format
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(validation_df),
    "test": Dataset.from_pandas(test_df)
})

# Display dataset information
print("\nDataset structure:")
print(dataset)
print("\nTrain set size:", len(dataset['train']))
print("Validation set size:", len(dataset['validation']))
print("Test set size:", len(dataset['test']))

# Show first example
print("\nFirst example from training set:")
print(dataset['train'][0])

# Show class information
train_labels = [l for l in dataset['train']['label'] if l is not None]
unique_labels = sorted(set(train_labels))
print("\nNumber of classes:", len(unique_labels))
print("Label range:", int(min(unique_labels)), "to", int(max(unique_labels)))

# Count data on each class
print("\n" + "="*50)
print("Class Distribution in Training Set:")
print("="*50)
train_label_counts = pd.Series(train_labels).value_counts().sort_index()
for label, count in train_label_counts.items():
    print(f"Label {int(label):3d}: {count:4d} samples")

print("\n" + "="*50)
print("Class Distribution in Validation Set:")
print("="*50)
validation_labels = [l for l in dataset['validation']['label'] if l is not None]
validation_label_counts = pd.Series(validation_labels).value_counts().sort_index()
for label, count in validation_label_counts.items():
    print(f"Label {int(label):3d}: {count:4d} samples")

print("\n" + "="*50)
print("Class Distribution in Test Set:")
print("="*50)
test_labels = [l for l in dataset['test']['label'] if l is not None]
test_label_counts = pd.Series(test_labels).value_counts().sort_index()
for label, count in test_label_counts.items():
    print(f"Label {int(label):3d}: {count:4d} samples")

print("\n" + "="*50)
print("Summary Statistics:")
print("="*50)
print(f"Train - Min samples per class: {train_label_counts.min()}")
print(f"Train - Max samples per class: {train_label_counts.max()}")
print(f"Train - Mean samples per class: {train_label_counts.mean():.2f}")
print(f"Train - Std samples per class: {train_label_counts.std():.2f}")
print(f"\nValidation - Min samples per class: {validation_label_counts.min()}")
print(f"Validation - Max samples per class: {validation_label_counts.max()}")
print(f"Validation - Mean samples per class: {validation_label_counts.mean():.2f}")
print(f"Validation - Std samples per class: {validation_label_counts.std():.2f}")
print(f"\nTest - Min samples per class: {test_label_counts.min()}")
print(f"Test - Max samples per class: {test_label_counts.max()}")
print(f"Test - Mean samples per class: {test_label_counts.mean():.2f}")
print(f"Test - Std samples per class: {test_label_counts.std():.2f}")

# Show some sample texts with their labels
print("\n" + "="*50)
print("Sample data (first 5 examples):")
print("="*50)
for i in range(5):
    print(f"  Label {int(dataset['train'][i]['label']):3d}: {dataset['train'][i]['utterance']}")

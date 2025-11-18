import pandas as pd
import os

# Load dataset from local CSV file
csv_file = "./twcs/twcs.csv"
# print(f"Loading dataset from: {csv_file}")

# Load the dataset
df = pd.read_csv(csv_file)

# Display basic information
print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

# print("\nFirst few rows:")
# print(df.head())

print("\nDataset info:")
print(df.info())

# Check for common columns and analyze
if 'author_id' in df.columns:
    print("\n" + "="*70)
    print("DISTRIBUSI DATA - JUMLAH ROW TIAP AUTHOR/COMPANY")
    print("="*70)
    author_counts = df['author_id'].value_counts()
    print(f"Total unique authors: {len(author_counts)}")
    print(f"Total tweets: {len(df)}")
    
    print("\n" + "="*70)
    print("TOP 20 MOST ACTIVE AUTHORS (Jumlah Tweet):")
    print("="*70)
    for rank, (author, count) in enumerate(author_counts.head(20).items(), 1):
        percentage = (count / len(df)) * 100
        print(f"{rank:2d}. {author:30s}: {count:6d} tweets ({percentage:5.2f}%)")
    
    print("\n" + "="*70)
    print("STATISTIK DISTRIBUSI:")
    print("="*70)
    print(f"Min tweets per author: {author_counts.min()}")
    print(f"Max tweets per author: {author_counts.max()}")
    print(f"Mean tweets per author: {author_counts.mean():.2f}")
    print(f"Median tweets per author: {author_counts.median():.2f}")
    print(f"Std tweets per author: {author_counts.std():.2f}")
    
    # Distribution of authors by tweet count
    # print("\n" + "="*70)
    # print("DISTRIBUSI AUTHOR BERDASARKAN JUMLAH TWEET:")
    # print("="*70)
    # bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
    # labels = ['1-10', '11-50', '51-100', '101-500', '501-1000', '1001-5000', '5000+']
    # distribution = pd.cut(author_counts, bins=bins, labels=labels).value_counts().sort_index()
    # for range_label, count in distribution.items():
    #     print(f"{range_label:12s}: {count:5d} authors")
    
    # print("\n" + "="*70)
    # print("ALL AUTHORS WITH TWEET COUNTS:")
    # print("="*70)
    # for rank, (author, count) in enumerate(author_counts.items(), 1):
    #     percentage = (count / len(df)) * 100
    #     print(f"{rank:3d}. {author:40s}: {count:6d} tweets ({percentage:5.2f}%)")

if 'inbound' in df.columns:
    print("\n" + "="*50)
    print("Inbound vs Outbound Distribution:")
    print("="*50)
    inbound_counts = df['inbound'].value_counts()
    print(inbound_counts)
    print(f"\nInbound (customer queries): {inbound_counts.get(True, 0)}")
    print(f"Outbound (company responses): {inbound_counts.get(False, 0)}")

if 'response_tweet_id' in df.columns:
    print("\n" + "="*50)
    print("Response Statistics:")
    print("="*50)
    has_response = df['response_tweet_id'].notna().sum()
    no_response = df['response_tweet_id'].isna().sum()
    print(f"Tweets with responses: {has_response}")
    print(f"Tweets without responses: {no_response}")

# Sample tweets
if 'text' in df.columns:
    print("\n" + "="*50)
    print("Sample Tweets (first 10):")
    print("="*50)
    for i, row in df.head(10).iterrows():
        text = row['text']
        inbound = row.get('inbound', 'N/A')
        direction = "INBOUND" if inbound == True else "OUTBOUND" if inbound == False else "N/A"
        print(f"{i+1}. [{direction}] {text[:80]}..." if len(str(text)) > 80 else f"{i+1}. [{direction}] {text}")

print("\n" + "="*50)
print("Missing Values:")
print("="*50)
print(df.isnull().sum())

print("\n" + "="*50)
print("Summary Statistics:")
print("="*50)
print(f"Total tweets: {len(df)}")
if 'author_id' in df.columns:
    print(f"Unique authors/companies: {df['author_id'].nunique()}")
if 'created_at' in df.columns:
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

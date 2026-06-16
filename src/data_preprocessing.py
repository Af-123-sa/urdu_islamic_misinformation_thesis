"""
Urdu Text Preprocessing Utilities
"""

import pandas as pd
import re

def clean_urdu_text(text):
    """Clean Urdu text by removing noise"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep Urdu letters)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def preprocess_dataset(csv_path, output_path):
    """Preprocess Urdu dataset"""
    df = pd.read_csv(csv_path)
    
    df['text'] = df['text'].apply(clean_urdu_text)
    
    # Remove empty texts
    df = df[df['text'].str.len() > 0]
    
    df.to_csv(output_path, index=False)
    print(f"✓ Preprocessed dataset saved to: {output_path}")
    print(f"✓ Total samples: {len(df)}")
    
    return df

if __name__ == "__main__":
    # Example usage
    preprocess_dataset("data/urdu_fake_news_dataset.csv", "data/preprocessed_dataset.csv")

"""
Thesis: Deep Learning-Based Detection of Fabricated Hadith and Fake Islamic Quotes
in Urdu Social Media Posts

Main Training Script
"""

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import os
from datetime import datetime

# ============================================
# CONFIGURATION
# ============================================

class Config:
    DATASET_PATH = "data/urdu_fake_news_dataset.csv"
    MODEL_NAME = "bert-base-multilingual-cased"  # or "Urmovt/urdu-bert"
    OUTPUT_DIR = "./results"
    MODEL_SAVE_DIR = "./models/urdu_islamic_model"
    MAX_LENGTH = 256
    BATCH_SIZE = 8
    NUM_EPOCHS = 5
    LEARNING_RATE = 2e-5
    
config = Config()

# Create output directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)

# ============================================
# STEP 1: LOAD DATASET
# ============================================

print("=" * 50)
print("STEP 1: Loading Dataset")
print("=" * 50)

try:
    df = pd.read_csv(config.DATASET_PATH)
    print(f"✓ Dataset loaded: {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    print(f"Label distribution:\n{df['label'].value_counts()}")
except FileNotFoundError:
    print("✗ Dataset not found! Creating sample dataset...")
    print("Download dataset from: https://github.com/MaazAmjad/Urdu-News-Augmented-Dataset")
    
    # Create sample dataset for testing
    df = pd.DataFrame({
        'text': [
            "صحیح بخاری میں ہے کہ رسول اللہ نے کہا",
            "یقین سے جانیں کہ یہ قول احادیث میں موجود ہے",
            "مسلم شریف میں ورذ ہے",
            " یہ حدیث غریب ہے اور کسی نے نہیں پڑھی",
        ],
        'label': [0, 1, 0, 1]  # 0=real, 1=fake
    })
    df.to_csv(config.DATASET_PATH, index=False)
    print(f"✓ Sample dataset created at {config.DATASET_PATH}")

# ============================================
# STEP 2: PREPARE DATA
# ============================================

print("\n" + "=" * 50)
print("STEP 2: Preparing Data")
print("=" * 50)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].tolist(), 
    df['label'].tolist(), 
    test_size=0.2, 
    random_state=42,
    stratify=df['label']
)

print(f"✓ Train samples: {len(train_texts)}")
print(f"✓ Test samples: {len(test_texts)}")

# ============================================
# STEP 3: LOAD TOKENIZER
# ============================================

print("\n" + "=" * 50)
print("STEP 3: Loading Tokenizer")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

train_encodings = tokenizer(
    train_texts, 
    truncation=True, 
    padding=True, 
    max_length=config.MAX_LENGTH
)
test_encodings = tokenizer(
    test_texts, 
    truncation=True, 
    padding=True, 
    max_length=config.MAX_LENGTH
)

print(f"✓ Tokenizer loaded: {config.MODEL_NAME}")
print(f"✓ Train tokenized: {len(train_encodings['input_ids'])} samples")

# ============================================
# STEP 4: CREATE DATASET CLASS
# ============================================

print("\n" + "=" * 50)
print("STEP 4: Creating Dataset Objects")
print("=" * 50)

class UrduFakeNewsDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

train_dataset = UrduFakeNewsDataset(train_encodings, train_labels)
test_dataset = UrduFakeNewsDataset(test_encodings, test_labels)

print(f"✓ Train dataset created: {len(train_dataset)} samples")
print(f"✓ Test dataset created: {len(test_dataset)} samples")

# ============================================
# STEP 5: LOAD MODEL
# ============================================

print("\n" + "=" * 50)
print("STEP 5: Loading Pre-trained Model")
print("=" * 50)

model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_NAME, 
    num_labels=2
)

print(f"✓ Model loaded: {config.MODEL_NAME}")
print(f"✓ Number of parameters: {model.num_parameters()}")

# ============================================
# STEP 6: DEFINE METRICS
# ============================================

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions),
        'precision': precision_score(labels, predictions),
        'recall': recall_score(labels, predictions)
    }

# ============================================
# STEP 7: TRAINING CONFIGURATION
# ============================================

print("\n" + "=" * 50)
print("STEP 6: Setting Training Parameters")
print("=" * 50)

training_args = TrainingArguments(
    output_dir=config.OUTPUT_DIR,
    num_train_epochs=config.NUM_EPOCHS,
    per_device_train_batch_size=config.BATCH_SIZE,
    per_device_eval_batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_dir=f"{config.OUTPUT_DIR}/logs",
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

print(f"✓ Training epochs: {config.NUM_EPOCHS}")
print(f"✓ Batch size: {config.BATCH_SIZE}")
print(f"✓ Learning rate: {config.LEARNING_RATE}")

# ============================================
# STEP 8: CREATE TRAINER
# ============================================

print("\n" + "=" * 50)
print("STEP 7: Creating Trainer")
print("=" * 50)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("✓ Trainer created successfully")

# ============================================
# STEP 9: TRAIN MODEL
# ============================================

print("\n" + "=" * 50)
print("STEP 8: Training Model")
print("=" * 50)

start_time = datetime.now()
trainer.train()
end_time = datetime.now()

print(f"✓ Training completed!")
print(f"⏱ Training time: {end_time - start_time}")

# ============================================
# STEP 10: EVALUATE MODEL
# ============================================

print("\n" + "=" * 50)
print("STEP 9: Evaluating Model")
print("=" * 50)

eval_results = trainer.evaluate()

print("\n📊 Evaluation Results:")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}")

# Save results to file
results_df = pd.DataFrame([eval_results])
results_df.to_csv(f"{config.OUTPUT_DIR}/evaluation_results.csv", index=False)

# ============================================
# STEP 11: SAVE MODEL
# ============================================

print("\n" + "=" * 50)
print("STEP 10: Saving Model")
print("=" * 50)

trainer.save_model(config.MODEL_SAVE_DIR)
tokenizer.save_pretrained(config.MODEL_SAVE_DIR)

print(f"✓ Model saved to: {config.MODEL_SAVE_DIR}")
print(f"✓ Tokenizer saved to: {config.MODEL_SAVE_DIR}")

# ============================================
# STEP 12: TEST ON NEW SAMPLES
# ============================================

print("\n" + "=" * 50)
print("STEP 11: Testing on New Samples")
print("=" * 50)

def predict_fake_islamic_quote(text):
    """Predict if an Urdu Islamic quote is fake or real"""
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=config.MAX_LENGTH
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, axis=1)
    
    return predictions.item()

# Test examples
test_samples = [
    "صحیح بخاری میں ہے کہ رسول اللہ نے کہا",  # Likely authentic
    "یقین سے جانیں کہ یہ قول احادیث میں موجود ہے",  # Likely fake
]

for sample in test_samples:
    result = predict_fake_islamic_quote(sample)
    prediction_label = '❌ FAKE' if result == 1 else '✅ REAL'
    print(f"\nText: {sample}")
    print(f"Prediction: {prediction_label}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"""
Model saved to: {config.MODEL_SAVE_DIR}
Results saved to: {config.OUTPUT_DIR}

Next steps:
1. Test on more Urdu Islamic quotes
2. Try different models (DistilBERT, XLM-RoBERTa)
3. Deploy as web API
4. Create visualization of results
""")

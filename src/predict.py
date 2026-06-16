"""
Prediction Script - Load trained model and predict on new Urdu text
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IslamicMisinformationPredictor:
    def __init__(self, model_path="./models/urdu_islamic_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
    
    def predict(self, text):
        """Predict if Urdu text is fake or real Islamic content"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=256
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, axis=1)
            probabilities = torch.softmax(outputs.logits, axis=1)
        
        label = 'FAKE' if predictions.item() == 1 else 'REAL'
        confidence = probabilities[0, predictions.item()].item()
        
        return {
            'label': label,
            'confidence': confidence,
            'prediction': predictions.item()
        }
    
    def predict_batch(self, texts):
        """Predict on multiple texts"""
        return [self.predict(text) for text in texts]

# Example usage
if __name__ == "__main__":
    predictor = IslamicMisinformationPredictor()
    
    test_texts = [
        "صحیح بخاری میں ہے کہ رسول اللہ نے کہا",
        "یقین سے جانیں کہ یہ قول احادیث میں موجود ہے",
    ]
    
    for text in test_texts:
        result = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")

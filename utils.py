import pandas as pd
import time
import random
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- CONFIGURATION ---
MODEL_REPO = "Razor2507/ComplaintsClassifier"

@st.cache_data
def load_global_analytics():
    data = pd.read_csv('subissueData.csv')
    return data

@st.cache_data
def load_geo_analytics():
    data = pd.read_csv('mapData.csv')
    return data

@st.cache_resource
def load_model():
    """
    Downloads the model architecture, weights, and tokenizer 
    from Hugging Face automatically.
    Cached so it runs only once.
    """
    print(f"Loading model from {MODEL_REPO}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
        
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Failed to load model from Hugging Face: {e}")
        return None, None, None

def predict_complaint_class(text):
    """
    The actual classification logic.
    """
    tokenizer, model, device = load_model()
    
    if model is None:
        return "Error loading model"

    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        padding=True, 
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=1).item()
    
    if hasattr(model.config, 'id2label') and model.config.id2label:
        return model.config.id2label[predicted_class_id]
    else:
        return f"Class {predicted_class_id}"
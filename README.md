# ZeroLedger - Banking Complaint Intelligence & Resolution System

ZeroLedger is an AI-powered Streamlit application that analyzes banking complaints, predicts the correct sub-issue using a fine-tuned RoBERTa model, and generates resolution guidance using a Groq-powered LLM. The dashboard also provides interactive analytics to explore complaint patterns across the US.

---

## Features

### AI Complaint Classification
- Fine-tuned RoBERTa-base model  
- Predicts 11 complaint sub-issues  
- Trained on 1.2M+ cleaned narratives  

### AI Resolution Assistant
- Accepts raw complaint text  
- Classifies the issue  
- Produces a personalized, policy-aligned resolution  

### Interactive Analytics Dashboard
- Top complaint categories  
- Timely-response insights  
- USA choropleth heatmap  
- State-wise breakdown of complaints and companies  

### Tech Stack
- **Frontend:** Streamlit, Plotly  
- **AI:** PyTorch, Transformers (RoBERTa), Groq API  
- **Big Data:** PySpark, MongoDB  
- **Deployment:** Streamlit Cloud + Secret Manager  

---

## Project Structure

ZeroLedger/
│
├── app.py # Main Streamlit UI
├── utils.py # Data loaders & helpers
├── llm_agent.py # Groq LLM resolution generator
├── data/ # Global & geo datasets
├── requirements.txt
└── README.md

---

## 🛠 Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ZeroLedger.git
cd ZeroLedger
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a file:
```bash
.streamlit/secrets.toml
```
Inside it:
```bash
[gcp]
GROQ_API_KEY = "your_key_here"
```

### 4. Run the application
```bash
streamlit run app.py
```

---

## Contributors

- Upayan Chaudhuri
- Aditya Chowdhury
- Sandip Shaw
- Vedant Vartak

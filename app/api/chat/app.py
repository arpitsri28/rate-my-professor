from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')

class TextData(BaseModel):
    text: str

@app.post("/embeddings/")
async def create_embeddings(data: TextData):
    try:
        # Prepare text data for model input
        inputs = tokenizer(data.text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        # Generate model output
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        return {"embeddings": embeddings}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


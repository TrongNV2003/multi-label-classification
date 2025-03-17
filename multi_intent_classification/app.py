from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import uvicorn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL = "bert-classification"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

unique_labels = ["Cung cấp thông tin", "Tương tác", "Hỏi thông tin giao hàng", 
                 "Hỗ trợ, hướng dẫn", "Yêu cầu", "Phản hồi", "Sự vụ", "UNKNOWN"]
id2label = {idx: label for idx, label in enumerate(unique_labels)}

# Input Schema
class PredictRequest(BaseModel):
    query: str
    history: List[str]

# Response Schema
class PredictResponse(BaseModel):
    query: str
    predicted_labels: List[str]
    history: List[str]

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict the intent of a query with optional conversation history.
    """
    model.eval()

    if request.history:
        history = request.history
        history_text = tokenizer.sep_token.join(history)
        input_text = f"<history>{history_text}</history><current>{request.query}</current>"
    else:
        input_text = f"<current>{request.query}</current>"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            preds = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()

        predicted_labels = [id2label[idx] for idx, val in enumerate(preds[0]) if val == 1.0]
        if not predicted_labels:
                predicted_labels.append("UNKNOWN")
        
        logger.info(f"Predicted Labels: {predicted_labels}")

        return PredictResponse(
            query=request.query,
            predicted_labels=predicted_labels,
            history=request.history if request.history else [],
        )
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    
if __name__ == "__main__":

    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=2206)
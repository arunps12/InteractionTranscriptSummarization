import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from src.interaction_transcript_summarization.logging import logger

# Initialize FastAPI app
app = FastAPI(
    title="Dialogue Summarization API",
    description="API for summarizing dialogue transcripts using fine-tuned PEGASUS model",
    version="1.0.0"
)

# Global variables for model and tokenizer
model = None
tokenizer = None
model_name = "google/pegasus-cnn_dailymail"
model_version = "1.0.0"
device = "cuda" if torch.cuda.is_available() else "cpu"


class SummarizeRequest(BaseModel):
    """Request model for summarization endpoint"""
    text: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it."
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for summarization endpoint"""
    summary: str
    model: str
    version: str
    latency_ms: float


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup"""
    global model, tokenizer, model_name
    
    try:
        # Try to load from local best model first
        model_path = "artifacts/model_trainer/model_best"
        
        if os.path.exists(model_path):
            logger.info(f"Loading model from local path: {model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_name = "pegasus-samsum-local"
        else:
            # Fallback to HuggingFace Hub or base model
            logger.info("Local model not found, loading base model")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
            tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
            model_name = "google/pegasus-cnn_dailymail"
        
        model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Dialogue Summarization API",
        "version": model_version,
        "model": model_name,
        "device": device,
        "endpoints": {
            "summarize": "/summarize",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """
    Summarize dialogue text.
    
    Args:
        request: SummarizeRequest containing the dialogue text
        
    Returns:
        SummarizeResponse with summary and metadata
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # Start timing
        start_time = time.time()
        
        # Tokenize input
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode summary
        summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(f"Summarization completed in {latency_ms:.2f}ms")
        
        return SummarizeResponse(
            summary=summary,
            model=model_name,
            version=model_version,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
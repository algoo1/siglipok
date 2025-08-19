import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import io
import base64
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SigLIP 2 Model API",
    description="API for SigLIP 2 So400m model inference",
    version="1.0.0"
)

# Global variables for model and processor
model = None
processor = None

@app.on_event("startup")
async def load_model():
    """Load the SigLIP 2 model and processor on startup"""
    global model, processor
    
    try:
        logger.info("Loading SigLIP 2 model...")
        model_name = "google/siglip2-so400m-patch14-384"
        
        # Load model and processor
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).eval()
        
        processor = AutoProcessor.from_pretrained(model_name)
        
        logger.info(f"Model loaded successfully on device: {next(model.parameters()).device}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "SigLIP 2 Model API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "processor_loaded": processor is not None,
        "cuda_available": torch.cuda.is_available(),
        "device": str(next(model.parameters()).device) if model else "N/A"
    }

@app.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    labels: str = "cat,dog,car,plane,person"
):
    """Classify an image with given labels"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Parse labels
        candidate_labels = [label.strip() for label in labels.split(',')]
        
        # Process image
        inputs = processor(
            images=[image],
            text=candidate_labels,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1)
        
        # Format results
        results = []
        for i, label in enumerate(candidate_labels):
            results.append({
                "label": label,
                "score": float(probs[0][i])
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "predictions": results,
            "top_prediction": results[0] if results else None
        }
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/embed")
async def get_image_embeddings(file: UploadFile = File(...)):
    """Get image embeddings from the vision encoder"""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Process image
        inputs = processor(images=[image], return_tensors="pt").to(model.device)
        
        # Get embeddings
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
        
        # Convert to list for JSON serialization
        embeddings_list = image_embeddings.cpu().numpy().tolist()
        
        return {
            "embeddings": embeddings_list,
            "shape": list(image_embeddings.shape),
            "dtype": str(image_embeddings.dtype)
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")

@app.post("/classify_url")
async def classify_image_url(
    image_url: str,
    labels: str = "cat,dog,car,plane,person"
):
    """Classify an image from URL with given labels"""
    try:
        # Load image from URL
        image = load_image(image_url)
        
        # Parse labels
        candidate_labels = [label.strip() for label in labels.split(',')]
        
        # Process image
        inputs = processor(
            images=[image],
            text=candidate_labels,
            return_tensors="pt",
            padding=True
        ).to(model.device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1)
        
        # Format results
        results = []
        for i, label in enumerate(candidate_labels):
            results.append({
                "label": label,
                "score": float(probs[0][i])
            })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "predictions": results,
            "top_prediction": results[0] if results else None
        }
        
    except Exception as e:
        logger.error(f"URL classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"URL classification failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
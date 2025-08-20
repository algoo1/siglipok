import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SigLIP 2 Model API", version="1.0.0")
model = None
processor = None

@app.on_event("startup")
async def load_model():
    global model, processor
    try:
        model_name = "google/siglip2-so400m-patch14-384"
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).eval()
        processor = AutoProcessor.from_pretrained(model_name)
        logger.info(f"Model loaded on {next(model.parameters()).device}")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise e

@app.get("/")
async def root():
    return {"message": "SigLIP 2 Model API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "SigLIP-2"}

def process_classify(image, labels):
    candidate_labels = [label.strip() for label in labels.split(',')]
    inputs = processor(images=[image], text=candidate_labels, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits_per_image, dim=-1)
    
    results = [{"label": label, "score": float(probs[0][i])} for i, label in enumerate(candidate_labels)]
    results.sort(key=lambda x: x["score"], reverse=True)
    return {"predictions": results, "top_prediction": results[0] if results else None}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...), labels: str = "cat,dog,car,plane,person"):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return process_classify(image, labels)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

@app.post("/embed")
async def get_image_embeddings(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        inputs = processor(images=[image], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)
        
        embeddings_list = image_embeddings.cpu().numpy().tolist()
        return {"embeddings": embeddings_list, "shape": list(image_embeddings.shape)}
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {e}")

@app.post("/classify_url")
async def classify_image_url(image_url: str, labels: str = "cat,dog,car,plane,person"):
    try:
        image = load_image(image_url)
        return process_classify(image, labels)
    except Exception as e:
        logger.error(f"URL classification error: {e}")
        raise HTTPException(status_code=500, detail=f"URL classification failed: {e}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
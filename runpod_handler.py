import runpod
import torch
import base64
import io
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor
import logging
import psutil
import gc
import time
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
processor = None

def load_model():
    global model, processor
    max_retries = 5
    model_name = "google/siglip-base-patch16-224"
    
    logger.info(f"Starting model loading process. CUDA available: {torch.cuda.is_available()}")
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model attempt {attempt + 1}/{max_retries}")
            
            # Load processor first
            logger.info("Loading processor...")
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            logger.info("Processor loaded successfully")
            
            # Load model
            logger.info("Loading model...")
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()
            
            if torch.cuda.is_available():
                logger.info("Moving model to GPU...")
                model = model.cuda()
                torch.cuda.empty_cache()
                logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            
            logger.info(f"Model loaded successfully on {next(model.parameters()).device}")
            return True
            
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}", exc_info=True)
            
            # Cleanup on failure
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 10  # Exponential backoff
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                logger.error("All model loading attempts failed")
                return False

def load_image_from_url(image_url):
    try:
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"URL load failed: {e}")
        return None

def decode_base64_image(base64_string):
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data)).convert('RGB')
    except Exception as e:
        logger.error(f"Base64 decode failed: {e}")
        return None

def classify_image(image, labels):
    try:
        candidate_labels = [label.strip() for label in labels.split(',') if label.strip()] if isinstance(labels, str) else [label for label in labels if label]
        if not candidate_labels:
            return {"error": "No valid labels provided"}
        
        inputs = processor(images=[image], text=candidate_labels, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()} if torch.cuda.is_available() else inputs.to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits_per_image, dim=-1).cpu().numpy()
        
        results = [{"label": label, "score": float(probs[0][i])} for i, label in enumerate(candidate_labels)]
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return {"predictions": results, "top_prediction": results[0] if results else None}
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"error": f"Classification failed: {e}"}

def get_image_embeddings(image):
    try:
        inputs = processor(images=[image], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()} if torch.cuda.is_available() else inputs.to(model.device)
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings_cpu = image_features.cpu().numpy()
            shape = list(image_features.shape)
        
        return {"task": "embed", "embeddings": embeddings_cpu.tolist(), "shape": shape}
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return {"error": f"Embedding failed: {e}"}

def handler(job):
    try:
        logger.info(f"Processing job: {job.get('id', 'unknown')}")
        
        # Check if model is loaded
        if model is None or processor is None:
            logger.error("Model or processor not loaded")
            return {"error": "Model not initialized. Please wait for model loading to complete."}
        
        if not job or "input" not in job:
            logger.error("Invalid job format received")
            return {"error": "Invalid job format"}
            
        job_input = job["input"]
        task_type = job_input.get("task", "classify")
        logger.info(f"Task type: {task_type}")
        
        image = None
        if "image_url" in job_input:
            image = load_image_from_url(job_input["image_url"])
        elif "image_base64" in job_input:
            image = decode_base64_image(job_input["image_base64"])
        elif "image_data" in job_input:
            image = decode_base64_image(job_input["image_data"])
        else:
            return {"error": "No image provided. Use 'image_url', 'image_base64', or 'image_data'"}
        
        if image is None:
            return {"error": "Failed to load image"}
        
        if task_type == "classify":
            result = classify_image(image, job_input.get("labels", "cat,dog,car,plane,person"))
        elif task_type in ["embed", "embed_image"]:
            result = get_image_embeddings(image)
        else:
            return {"error": f"Unknown task type: {task_type}. Use 'classify' or 'embed'"}
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Job completed successfully with task: {task_type}")
        return result
            
    except Exception as e:
        logger.error(f"Handler error: {e}", exc_info=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"error": f"Processing failed: {str(e)}"}

def get_status():
    try:
        status = {
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "device": str(next(model.parameters()).device) if model else "unknown"
        }
        if torch.cuda.is_available():
            status.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved()
            })
        return status
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    for attempt in range(3):
        if load_model():
            break
        elif attempt < 2:
            import time
            time.sleep(5)
    else:
        raise Exception("Model initialization failed")
    
    runpod.serverless.start({"handler": handler})
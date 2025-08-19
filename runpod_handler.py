import runpod
import torch
import base64
import io
from PIL import Image
from transformers import AutoModel, AutoProcessor
from transformers.image_utils import load_image
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load the SigLIP 2 model and processor"""
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
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

def decode_base64_image(base64_string):
    """Decode base64 image string to PIL Image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        return None

def classify_image(image, labels):
    """Classify image with given labels"""
    try:
        # Parse labels
        if isinstance(labels, str):
            candidate_labels = [label.strip() for label in labels.split(',')]
        else:
            candidate_labels = labels
        
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
        raise e

def get_image_embeddings(image):
    """Get image embeddings from the vision encoder"""
    try:
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
        raise e

def handler(job):
    """RunPod handler function"""
    try:
        job_input = job["input"]
        task_type = job_input.get("task", "classify")
        
        # Load image
        image = None
        if "image_url" in job_input:
            image = load_image(job_input["image_url"])
        elif "image_base64" in job_input:
            image = decode_base64_image(job_input["image_base64"])
        else:
            return {"error": "No image provided. Use 'image_url' or 'image_base64'"}
        
        if image is None:
            return {"error": "Failed to load image"}
        
        # Process based on task type
        if task_type == "classify":
            labels = job_input.get("labels", "cat,dog,car,plane,person")
            result = classify_image(image, labels)
            return result
            
        elif task_type == "embed":
            result = get_image_embeddings(image)
            return result
            
        else:
            return {"error": f"Unknown task type: {task_type}"}
            
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Load model on startup
    if load_model():
        logger.info("Starting RunPod serverless handler...")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load model, exiting...")
        exit(1)
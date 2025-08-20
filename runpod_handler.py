import runpod
import torch
import base64
import io
import requests
from PIL import Image
from transformers import AutoModel, AutoProcessor
import logging
import gc
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and processor
model = None
processor = None

def load_model():
    """Load the SigLIP 2 model and processor with optimizations"""
    global model, processor
    
    try:
        logger.info("Loading SigLIP 2 model...")
        model_name = "google/siglip-base-patch16-224"
        
        # Load processor first (faster)
        logger.info("Loading processor...")
        processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model with basic optimizations
        logger.info("Loading model...")
        model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        ).eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
        
        logger.info(f"Model loaded successfully on device: {next(model.parameters()).device}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Clean up on failure
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return False

def load_image_from_url(image_url):
    """Load image from URL - optimized"""
    try:
        response = requests.get(image_url, timeout=30, stream=True)
        response.raise_for_status()
        
        # Load image with memory optimization
        image = Image.open(io.BytesIO(response.content))
        rgb_image = image.convert("RGB")
        
        # Clean up
        image.close()
        del response
        gc.collect()
        
        return rgb_image
    except Exception as e:
        logger.error(f"Failed to load image from URL: {str(e)}")
        return None

def decode_base64_image(base64_string):
    """Decode base64 image - optimized"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        # Decode with memory optimization
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        rgb_image = image.convert('RGB')
        
        # Clean up
        image.close()
        del image_data
        gc.collect()
        
        return rgb_image
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        return None

def classify_image(image, labels):
    """Classify image with given labels - optimized"""
    try:
        # Parse labels efficiently
        if isinstance(labels, str):
            candidate_labels = [label.strip() for label in labels.split(',') if label.strip()]
        else:
            candidate_labels = [label for label in labels if label]
        
        if not candidate_labels:
            return {"error": "No valid labels provided"}
        
        # Process image with memory optimization
        inputs = processor(
            images=[image],
            text=candidate_labels,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device efficiently
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(model.device)
        
        # Run inference with memory efficiency
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1)
            
            # Convert to CPU immediately to free GPU memory
            probs_cpu = probs.cpu().numpy()
        
        # Format results efficiently
        results = [
            {
                "label": label,
                "score": float(probs_cpu[0][i])
            }
            for i, label in enumerate(candidate_labels)
        ]
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Clean up tensors
        del inputs, outputs, logits_per_image, probs, probs_cpu
        
        return {
            "predictions": results,
            "top_prediction": results[0] if results else None
        }
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        raise e

def get_image_embeddings(image):
    """Get image embeddings - optimized"""
    try:
        # Process image with memory optimization
        inputs = processor(images=[image], return_tensors="pt")
        
        # Move to device efficiently
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(model.device)
        
        # Get embeddings with memory efficiency
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Normalize embeddings
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Convert to CPU immediately
            embeddings_cpu = image_features.cpu().numpy()
            shape = list(image_features.shape)
        
        # Clean up tensors
        del inputs, image_features
        
        return {
            "task": "embed",
            "embeddings": embeddings_cpu.tolist(),
            "shape": shape
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        return {"error": str(e)}

def handler(job):
    """RunPod handler function with optimizations"""
    try:
        # Quick validation
        if not job or "input" not in job:
            return {"error": "Invalid job format"}
            
        job_input = job["input"]
        task_type = job_input.get("task", "classify")
        
        # Load image efficiently
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
        
        # Process based on task type
        result = None
        if task_type == "classify":
            labels = job_input.get("labels", "cat,dog,car,plane,person")
            result = classify_image(image, labels)
            
        elif task_type == "embed" or task_type == "embed_image":
            result = get_image_embeddings(image)
            
        else:
            return {"error": f"Unknown task type: {task_type}. Use 'classify' or 'embed'"}
        
        # Clean up memory
        del image
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return result
            
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        # Clean up on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"error": str(e)}

def get_status():
    """Get model status and memory info"""
    try:
        status = {
            "model_loaded": model is not None,
            "processor_loaded": processor is not None,
            "device": str(next(model.parameters()).device) if model else "unknown"
        }
        
        if torch.cuda.is_available():
            status.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved(),
                "gpu_memory_free": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            })
        
        return status
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Initialize model on startup with retry
    max_retries = 3
    model_loaded = False
    
    for attempt in range(max_retries):
        if load_model():
            logger.info(f"Model loaded successfully on attempt {attempt + 1}")
            model_loaded = True
            break
        else:
            logger.warning(f"Failed to load model on attempt {attempt + 1}")
            if attempt < max_retries - 1:
                logger.info("Retrying in 5 seconds...")
                import time
                time.sleep(5)
    
    if not model_loaded:
        logger.error("Failed to load model after all retries")
        raise Exception("Model initialization failed")

    # Start the RunPod serverless function
    runpod.serverless.start({"handler": handler})
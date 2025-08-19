# SigLIP 2 RunPod Deployment

This repository contains everything needed to deploy the [SigLIP 2 So400m model](https://huggingface.co/google/siglip2-so400m-patch14-384) on RunPod for image classification and embedding extraction.

## Features

- **Zero-shot image classification** with custom labels
- **Image embedding extraction** using the vision encoder
- **FastAPI web service** for HTTP endpoints
- **RunPod serverless** integration
- **GPU acceleration** support
- **Docker containerization** for easy deployment

## Quick Start

### 1. Clone this repository

```bash
git clone https://github.com/algoo1/siglipok
cd siglip2-runpod
```

### 2. Build Docker image

```bash
docker build -t siglip2-runpod .
```

### 3. Run locally (optional)

```bash
docker run -p 8000:8000 --gpus all siglip2-runpod
```

### 4. Deploy to RunPod

1. Push your Docker image to Docker Hub:
   ```bash
   docker tag siglip2-runpod algonum1/siglip2-runpod:latest
   docker push algonum1/siglip2-runpod:latest
   ```

2. Create a new RunPod template:
   - Go to RunPod dashboard
   - Create new template
   - Use your Docker image: `algonum1/siglip2-runpod:latest`
   - Set container disk to 50GB+
   - Expose port 8000
   - Add environment variables if needed

3. Deploy as serverless endpoint or persistent pod

## API Endpoints

### Health Check
```bash
GET /health
```

### Image Classification
```bash
POST /classify
```

**Parameters:**
- `file`: Image file (multipart/form-data)
- `labels`: Comma-separated labels (default: "cat,dog,car,plane,person")

**Example:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -F "file=@image.jpg" \
  -F "labels=cat,dog,bird,car"
```

### Image Classification (URL)
```bash
POST /classify_url
```

**Parameters:**
- `image_url`: URL of the image
- `labels`: Comma-separated labels

**Example:**
```bash
curl -X POST "http://localhost:8000/classify_url" \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "labels": "cat,dog,bird,car"
  }'
```

### Image Embeddings
```bash
POST /embed
```

**Parameters:**
- `file`: Image file (multipart/form-data)

**Example:**
```bash
curl -X POST "http://localhost:8000/embed" \
  -F "file=@image.jpg"
```

## RunPod Serverless Usage

For serverless deployment, use the RunPod handler:

### Classification Example
```python
import runpod

# Initialize RunPod client
runpod.api_key = "YOUR_RUNPOD_API_KEY"  # Replace with your actual RunPod API key

# Run classification (replace ENDPOINT_ID with your actual endpoint ID after creating it)
result = runpod.run(
    endpoint_id="ENDPOINT_ID",
    job_input={
        "task": "classify",
        "image_url": "https://example.com/image.jpg",
        "labels": "cat,dog,bird,car"
    }
)

print(result)
```

### Embedding Example
```python
result = runpod.run(
    endpoint_id="ENDPOINT_ID",
    job_input={
        "task": "embed",
        "image_url": "https://example.com/image.jpg"
    }
)

print(f"Embedding shape: {result['shape']}")
```

## Configuration

### Environment Variables

- `TRANSFORMERS_CACHE`: Cache directory for Hugging Face models
- `HF_HOME`: Hugging Face home directory
- `TORCH_HOME`: PyTorch cache directory
- `CUDA_VISIBLE_DEVICES`: GPU device selection

### Resource Requirements

**Minimum:**
- GPU: NVIDIA RTX A6000 or better
- RAM: 16GB
- Storage: 50GB
- CPU: 4 cores

**Recommended:**
- GPU: NVIDIA A100
- RAM: 32GB
- Storage: 100GB
- CPU: 8 cores

## Model Information

This deployment uses the **SigLIP 2 So400m** model:
- **Model**: `google/siglip2-so400m-patch14-384`
- **Input Resolution**: 384x384
- **Parameters**: ~400M
- **Capabilities**: Zero-shot classification, image-text retrieval, embedding extraction

## Performance

- **Inference Time**: ~100-200ms per image (A100 GPU)
- **Throughput**: ~5-10 images/second
- **Memory Usage**: ~8-12GB GPU memory

## Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size or use smaller model
   - Ensure sufficient GPU memory (8GB+)

2. **Model Download Timeout**
   - Pre-download model in Dockerfile
   - Use persistent storage for model cache

3. **Slow Inference**
   - Ensure GPU is being used
   - Check CUDA installation
   - Use appropriate data types (float16)

### Logs

Check container logs for debugging:
```bash
docker logs CONTAINER_ID
```

## Development

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

3. Test endpoints:
   ```bash
   curl http://localhost:8000/health
   ```

### Testing

Test the API with sample images:

```python
import requests

# Test classification
with open('test_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f},
        data={'labels': 'cat,dog,bird'}
    )
    print(response.json())
```

## License

This project is licensed under the MIT License. The SigLIP 2 model is subject to its own license terms.

## Citation

If you use this deployment, please cite the original SigLIP 2 paper:

```bibtex
@misc{tschannen2025siglip2multilingualvisionlanguage,
      title={SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features}, 
      author={Michael Tschannen and Alexey Gritsenko and Xiao Wang and Muhammad Ferjad Naeem and Ibrahim Alabdulmohsin and Nikhil Parthasarathy and Talfan Evans and Lucas Beyer and Ye Xia and Basil Mustafa and Olivier Hénaff and Jeremiah Harmsen and Andreas Steiner and Xiaohua Zhai},
      year={2025},
      eprint={2502.14786},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.14786}, 
}
```

## Support

For issues and questions:
- Create an issue in this repository
- Check RunPod documentation
- Review Hugging Face model page
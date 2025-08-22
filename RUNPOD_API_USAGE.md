# RunPod SigLIP API Usage Guide

This guide demonstrates how to use the SigLIP model deployed on RunPod for image classification and embedding tasks.

## API Endpoint

```
POST https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run
```

## Authentication

Include your RunPod API key in the Authorization header:

```
Authorization: Bearer YOUR_RUNPOD_API_KEY
```

## Image Classification

### Request Format

```json
{
  "input": {
    "task": "classify",
    "image_url": "https://example.com/image.jpg",
    "labels": "cat,dog,bird,car,person"
  }
}
```

### PowerShell Example

```powershell
Invoke-WebRequest -Uri "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" `
  -Method POST `
  -Headers @{
    "Authorization" = "Bearer YOUR_RUNPOD_API_KEY"
    "Content-Type" = "application/json"
  } `
  -Body '{
    "input": {
      "task": "classify",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/290px-Cat_November_2010-1a.jpg",
      "labels": "cat,dog,bird,car,person"
    }
  }'
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "classify",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/290px-Cat_November_2010-1a.jpg",
      "labels": "cat,dog,bird,car,person"
    }
  }'
```

## Image Embedding

### Request Format

```json
{
  "input": {
    "task": "embed",
    "image_url": "https://example.com/image.jpg"
  }
}
```

### PowerShell Example

```powershell
Invoke-WebRequest -Uri "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" `
  -Method POST `
  -Headers @{
    "Authorization" = "Bearer YOUR_RUNPOD_API_KEY"
    "Content-Type" = "application/json"
  } `
  -Body '{
    "input": {
      "task": "embed",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/290px-Cat_November_2010-1a.jpg"
    }
  }'
```

### cURL Example

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "embed",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/290px-Cat_November_2010-1a.jpg"
    }
  }'
```

## Using Base64 Images

You can also send images as base64-encoded strings:

```json
{
  "input": {
    "task": "classify",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ...",
    "labels": "cat,dog,bird,car,person"
  }
}
```

## Response Format

All requests return a job ID and status:

```json
{
  "id": "3ee14ebd-6f89-4ba5-9392-fe2ea0bb97f6-e2",
  "status": "IN_QUEUE"
}
```

## Supported Input Parameters

### Classification Task
- `task`: "classify" (required)
- `image_url`: URL to the image (optional if image_base64 provided)
- `image_base64`: Base64-encoded image (optional if image_url provided)
- `labels`: Comma-separated list of labels (default: "cat,dog,car,plane,person")

### Embedding Task
- `task`: "embed" (required)
- `image_url`: URL to the image (optional if image_base64 provided)
- `image_base64`: Base64-encoded image (optional if image_url provided)

## Model Information

- **Model**: Google SigLIP (siglip-base-patch16-224)
- **GPU**: NVIDIA RTX A6000
- **Memory**: 16GB
- **Storage**: 50GB
- **Max Workers**: 1
- **Execution Timeout**: 120 seconds

## Error Handling

The API includes comprehensive error handling for:
- Invalid image URLs
- Malformed base64 images
- Missing required parameters
- Model inference errors
- GPU memory issues

## Performance Notes

- Images are automatically resized to 224x224 pixels
- The model uses float16 precision on GPU for faster inference
- GPU memory is automatically cleared after each request
- Supports both CUDA and CPU inference
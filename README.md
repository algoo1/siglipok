# SigLIP 2 RunPod Serverless API

Optimized serverless implementation of Google's SigLIP 2 model for image classification and embedding generation on RunPod.

## Features

- **Image Classification**: Classify images with custom labels
- **Image Embeddings**: Generate normalized image embeddings
- **Memory Optimized**: Efficient GPU memory management
- **Model Warming**: Pre-warmed model for faster inference
- **Auto Retry**: Automatic model loading with retry mechanism
- **Performance Monitoring**: Built-in status and memory monitoring

## API Endpoints

### Image Classification
```json
{
  "input": {
    "task": "classify",
    "image_url": "https://example.com/image.jpg",
    "labels": "cat,dog,car,plane,person"
  }
}
```

### Image Embeddings
```json
{
  "input": {
    "task": "embed",
    "image_base64": "data:image/jpeg;base64,/9j/4AAQ..."
  }
}
```

## Performance Optimizations

- **Model Warming**: Dummy inference on startup for faster first request
- **Memory Management**: Automatic GPU memory cleanup after each request
- **Efficient Loading**: Optimized model loading with `low_cpu_mem_usage=True`
- **Tensor Cleanup**: Immediate CPU conversion and tensor deletion
- **Garbage Collection**: Automatic memory cleanup

## Deployment

1. Push code to GitHub repository
2. Create new release tag (e.g., `v1.2.0`)
3. RunPod will automatically build and deploy
4. Use the generated endpoint for inference

## Model Details

- **Model**: `google/siglip2-so400m-patch14-384`
- **Input Size**: 384x384 pixels
- **Precision**: FP16 on GPU, FP32 on CPU
- **Device**: Auto-detected (CUDA if available)

## Error Handling

- Comprehensive error logging
- Graceful fallbacks for memory issues
- Automatic retry mechanism for model loading
- Input validation and sanitization

## Memory Usage

- Optimized for RTX A5000 (24GB VRAM)
- Automatic memory cleanup between requests
- GPU memory monitoring and reporting
- Efficient batch processing support

## Recent Updates (v1.2.0)

- Added model warming for faster cold starts
- Implemented comprehensive memory management
- Enhanced error handling and retry logic
- Optimized image loading and processing
- Added performance monitoring capabilities
- Improved tensor cleanup and garbage collection
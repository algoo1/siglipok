# دليل إعادة نشر RunPod

## الخطوات السريعة لإعادة النشر

### 1. إيقاف الخدمة الحالية
```bash
# في لوحة تحكم RunPod
# اذهب إلى Serverless > Endpoints
# اختر الخدمة الحالية
# اضغط على "Stop" أو "Delete"
```

### 2. إنشاء خدمة جديدة
```bash
# في لوحة تحكم RunPod
# اضغط على "+ New Endpoint"
# اختر Template أو Custom
```

### 3. إعدادات النشر المطلوبة

#### أ. إعدادات الحاوية (Container Settings):
```yaml
Container Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04
Container Disk: 20 GB (minimum)
Container Registry Credentials: (if needed)
```

#### ب. إعدادات البيئة (Environment Variables):
```bash
TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface
HF_HOME=/runpod-volume/.cache/huggingface
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
```

#### ج. إعدادات الشبكة (Network Settings):
```yaml
HTTP Port: 8000
HTTP Path: /
Advanced: Enable
```

#### د. إعدادات الموارد (Resource Settings):
```yaml
GPU: RTX 4090 (24GB) أو أعلى
CPU: 8 vCPU (minimum)
RAM: 32 GB (minimum)
Storage: 50 GB (recommended)
```

### 4. رفع الملفات

#### الطريقة الأولى: GitHub Integration
```bash
# في إعدادات النشر
# اختر "GitHub" كمصدر
# أدخل: https://github.com/algoo1/siglipok.git
# Branch: main
```

#### الطريقة الثانية: رفع مباشر
```bash
# ارفع الملفات التالية:
- runpod_handler.py
- runpod_config.yaml
- requirements.txt
- Dockerfile (اختياري)
```

### 5. إعدادات التشغيل

#### أ. Handler Function:
```python
Handler: runpod_handler.handler
```

#### ب. Startup Command:
```bash
pip install -r requirements.txt && python runpod_handler.py
```

#### ج. Health Check:
```bash
# URL: /health
# Method: GET
# Timeout: 30 seconds
```

### 6. مراقبة النشر

#### أ. فحص السجلات:
```bash
# في لوحة التحكم
# اذهب إلى "Logs" tab
# راقب رسائل تحميل النموذج
# تأكد من عدم وجود أخطاء
```

#### ب. اختبار الخدمة:
```bash
# انتظر حتى تظهر حالة "Running"
# اختبر endpoint باستخدام:
curl -X POST "https://api.runpod.ai/v2/YOUR_NEW_ENDPOINT_ID/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"task":"classify","image_url":"https://example.com/test.jpg","labels":["test"]}}'
```

### 7. استكشاف الأخطاء

#### إذا فشل النشر:
1. **فحص السجلات**: ابحث عن رسائل الخطأ
2. **فحص الموارد**: تأكد من كفاية GPU memory
3. **فحص التبعيات**: تأكد من تثبيت جميع المكتبات
4. **فحص التكوين**: راجع runpod_config.yaml

#### إذا بقيت الطلبات في IN_QUEUE:
1. **انتظر 5-10 دقائق**: قد يحتاج النموذج وقت للتحميل
2. **فحص memory usage**: تأكد من عدم نفاد الذاكرة
3. **إعادة تشغيل**: أعد تشغيل الخدمة

### 8. نصائح للأداء الأمثل

```yaml
# في runpod_config.yaml
max_workers: 2
idle_timeout: 600
execution_timeout: 300
startup_timeout: 600

# استخدم GPU قوية
GPU: RTX 4090 24GB أو A100 40GB

# تأكد من التخزين المؤقت
volume_mount_path: /runpod-volume
```

## الملفات المطلوبة للنشر

1. **runpod_handler.py** - معالج الطلبات الرئيسي
2. **runpod_config.yaml** - إعدادات الخدمة
3. **requirements.txt** - التبعيات المطلوبة
4. **Dockerfile** - (اختياري) إعدادات الحاوية

## روابط مفيدة

- [RunPod Documentation](https://docs.runpod.io/)
- [RunPod Console](https://www.runpod.io/console)
- [GitHub Repository](https://github.com/algoo1/siglipok)
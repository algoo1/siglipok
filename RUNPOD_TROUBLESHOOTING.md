# RunPod Troubleshooting Guide

## مشكلة بقاء الطلبات في حالة IN_QUEUE

### الأسباب المحتملة:

1. **فشل تحميل النموذج**: النموذج قد يفشل في التحميل مما يؤدي لتوقف المعالج
2. **مهلة زمنية قصيرة**: مهلة التنفيذ قد تكون قصيرة جداً لتحميل النموذج الكبير
3. **عدد العمال المحدود**: worker واحد فقط يسبب اختناق في المعالجة
4. **مشاكل في الذاكرة**: نفاد ذاكرة GPU أثناء تحميل النموذج

### الحلول المطبقة:

#### 1. تحديث إعدادات RunPod (`runpod_config.yaml`):
```yaml
serverless:
  handler: "runpod_handler.py"
  max_workers: 2              # زيادة عدد العمال
  idle_timeout: 600           # زيادة مهلة الخمول
  execution_timeout: 300      # زيادة مهلة التنفيذ
  startup_timeout: 600        # إضافة مهلة بدء التشغيل
```

#### 2. تحسين معالج RunPod (`runpod_handler.py`):
- إضافة فحص حالة النموذج قبل المعالجة
- تحسين تسجيل الأخطاء والأحداث
- زيادة عدد محاولات تحميل النموذج إلى 5
- إضافة انتظار تدريجي بين المحاولات
- تحسين إدارة ذاكرة GPU

#### 3. تحديث Dockerfile:
- تحسين متغيرات البيئة
- إضافة مسارات التخزين المؤقت الصحيحة
- تحسين فحص الصحة
- إضافة فترة بدء للفحص الصحي

### خطوات إعادة النشر:

#### 1. بناء صورة Docker جديدة:
```bash
docker build -t algonum1/siglip2-runpod:latest .
docker push algonum1/siglip2-runpod:latest
```

#### 2. تحديث النشر على RunPod:
- قم بتسجيل الدخول إلى لوحة تحكم RunPod
- انتقل إلى قسم Serverless
- ابحث عن النشر `siglip2-so400m-deployment`
- اضغط على "Update" أو "Redeploy"
- تأكد من استخدام الصورة المحدثة

#### 3. مراقبة السجلات:
```bash
# عرض سجلات النشر
runpod logs <deployment-id>

# مراقبة السجلات في الوقت الفعلي
runpod logs <deployment-id> --follow
```

### اختبار الإصلاحات:

#### 1. اختبار تصنيف الصور:
```bash
curl -X POST "https://api.runpod.ai/v2/y7flbj7woig39y/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "classify",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/290px-Cat_November_2010-1a.jpg",
      "labels": "cat,dog,bird,car,person"
    }
  }'
```

#### 2. اختبار استخراج التضمينات:
```bash
curl -X POST "https://api.runpod.ai/v2/y7flbj7woig39y/run" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "embed",
      "image_url": "https://upload.wikimedia.org/wikipedia/commons/e/e3/Kheops-Pyramid.jpg"
    }
  }'
```

### مراقبة الأداء:

#### مؤشرات مهمة للمراقبة:
- **وقت بدء التشغيل**: يجب أن يكون أقل من 10 دقائق
- **استخدام الذاكرة**: مراقبة استخدام GPU وRAM
- **معدل النجاح**: نسبة الطلبات المكتملة بنجاح
- **وقت الاستجابة**: متوسط وقت معالجة الطلب

#### أوامر مفيدة للمراقبة:
```bash
# فحص حالة النشر
runpod status <deployment-id>

# عرض إحصائيات الاستخدام
runpod metrics <deployment-id>

# فحص صحة النشر
runpod health <deployment-id>
```

### استكشاف الأخطاء الشائعة:

#### 1. خطأ "Model not initialized":
- تحقق من سجلات تحميل النموذج
- تأكد من توفر مساحة كافية في الذاكرة
- زيادة مهلة بدء التشغيل

#### 2. خطأ "CUDA out of memory":
- تقليل حجم الدفعة
- استخدام precision أقل (float16)
- تنظيف ذاكرة GPU بانتظام

#### 3. خطأ "Connection timeout":
- زيادة مهلة الشبكة
- التحقق من صحة URL الصورة
- استخدام صور أصغر حجماً

### نصائح للأداء الأمثل:

1. **استخدام التخزين المؤقت**: تأكد من تكوين مسارات التخزين المؤقت بشكل صحيح
2. **تحسين الصور**: ضغط الصور قبل الإرسال
3. **مراقبة الموارد**: مراقبة استخدام CPU وGPU بانتظام
4. **التحديث المنتظم**: تحديث المكتبات والنموذج بانتظام

### الدعم والمساعدة:

- **وثائق RunPod**: https://docs.runpod.io/
- **مجتمع RunPod**: https://discord.gg/runpod
- **دعم فني**: support@runpod.io
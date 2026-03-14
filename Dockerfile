FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.4.0+cpu torchvision==0.19.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    ultralytics==8.4.21 \
    opencv-python-headless \
    grad-cam numpy

WORKDIR /app

COPY attention_module/  ./attention_module/
COPY models/            ./models/
COPY scripts/           ./scripts/
COPY results/           ./results/
COPY test_images/       ./test_images/
COPY yolov8_cbam.yaml   ./
COPY data.yaml          ./

RUN python -c "\
import ultralytics, os, shutil; \
src = '/app/attention_module/cbam.py'; \
dst_dir = os.path.join(os.path.dirname(ultralytics.__file__), 'nn', 'modules'); \
os.makedirs(dst_dir, exist_ok=True); \
dst = os.path.join(dst_dir, 'attention.py'); \
shutil.copy(src, dst); \
print('CBAM registered at:', dst) \
"

RUN python -c "from ultralytics import YOLO; YOLO('models/best.pt'); print('baseline OK')"
RUN python -c "from ultralytics import YOLO; YOLO('models/best_cbam.pt'); print('CBAM OK')"

CMD ["python", "scripts/demo.py", "test_images"]

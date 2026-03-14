from ultralytics import YOLO
import time
from glob import glob
import sys

weights = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
imgs = sorted(glob("valid/images/*.jpg"))[:20]  # small test set
if not imgs:
    print("No test images found in valid/images/")
    sys.exit(1)

model = YOLO(weights)
model.to("cpu")

# warmup
for _ in range(5):
    _ = model(imgs[0], device="cpu", imgsz=640, verbose=False)

t0 = time.time()
for im in imgs:
    _ = model(im, device="cpu", imgsz=640, verbose=False)
dt = time.time() - t0

print(f"{weights}: avg {dt/len(imgs)*1000:.1f} ms/img on CPU")

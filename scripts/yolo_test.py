
from ultralytics import YOLO
import os
from pathlib import Path


MODEL_PATH = input("Enter path to YOLOv8 model (.pt file): ").strip()
IMAGES_FOLDER = input("Enter path to folder containing images: ").strip()
RESULTS_FOLDER = input("Enter path to save predictions: ").strip()

# Device: -1 = CPU
DEVICE = -1


SAVE_TXT = True

# ── SETUP ──
os.makedirs(RESULTS_FOLDER, exist_ok=True)


image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
image_files = [str(p) for p in Path(IMAGES_FOLDER).glob("*") if p.suffix.lower() in image_extensions]

if len(image_files) == 0:
    print("No images found in the folder:", IMAGES_FOLDER)
    exit()

print(f"Found {len(image_files)} images. Starting prediction...")

model = YOLO(MODEL_PATH)
print("Model loaded successfully.")

# ── PREDICTION ──────
for img_path in image_files:
    print(f"Predicting: {img_path}")
    
    results = model.predict(
        source=img_path,
        device="cpu",
        save=True,                 
        save_txt=SAVE_TXT,         
        project=RESULTS_FOLDER,    
        exist_ok=True              
    )

print("Predictions completed. Check the folder:", RESULTS_FOLDER)
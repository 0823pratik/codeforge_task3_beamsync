#!/usr/bin/env python3
"""
Track 3 Demo: Runs both baseline + CBAM detection AND Grad-CAM
Usage: python scripts/demo.py [test_images_folder]
"""
import sys, os, json, time
import torch, cv2, numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

MODELS = {
    "baseline": "models/best.pt",
    "cbam":     "models/best_cbam.pt",
}
OUTPUT_DIRS = {
    "baseline": "test_images_yolo",
    "cbam":     "test_images_cbam",
    "heatmap_baseline": "test_images_heatmap",
    "heatmap_cbam":     "test_images_heatmap_cbam",
}

class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out[:, 4:, :].mean(-1)

def draw_boxes(img_bgr_640, results, names):
    h0, w0 = results.orig_shape
    out = img_bgr_640.copy()
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1 * 640 / w0); y1 = int(y1 * 640 / h0)
        x2 = int(x2 * 640 / w0); y2 = int(y2 * 640 / h0)
        label = f"{names[int(box.cls[0])]} {box.conf[0]:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, label, (x1, max(y1-6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return out

def run_gradcam(yolo, img_rgb_float, results, out_path):
    tensor  = torch.from_numpy(img_rgb_float).permute(2, 0, 1).unsqueeze(0)
    wrapped = YOLOWrapper(yolo.model)
    cam     = EigenCAM(model=wrapped, target_layers=[yolo.model.model[9]])
    gcam    = cam(input_tensor=tensor)[0, :]
    cam_img = show_cam_on_image(img_rgb_float, gcam, use_rgb=True)

    h0, w0 = results.orig_shape
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = int(x1*640/w0); y1 = int(y1*640/h0)
        x2 = int(x2*640/w0); y2 = int(y2*640/h0)
        label = f"{yolo.names[int(box.cls[0])]} {box.conf[0]:.2f}"
        cv2.rectangle(cam_img, (x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(cam_img, label, (x1, max(y1-6,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,0,0), 2)

    cv2.imwrite(str(out_path), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

def process_folder(images_folder):
    images_folder = Path(images_folder)
    image_files   = sorted(images_folder.glob("*"))
    image_files   = [p for p in image_files if p.suffix.lower() in (".jpg",".jpeg",".png",".bmp")]

    if not image_files:
        print(f"No images found in {images_folder}"); return

    for d in OUTPUT_DIRS.values():
        os.makedirs(d, exist_ok=True)

    summary = {}

    for tag, weights in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {tag.upper()} → {weights}")
        yolo = YOLO(weights); yolo.to("cpu")

        det_dir  = Path(OUTPUT_DIRS[tag])
        gcam_dir = Path(OUTPUT_DIRS[f"heatmap_{tag}"])

        t_total = 0
        for img_path in image_files:
            print(f"  {img_path.name}", end=" ... ")

            img_bgr  = cv2.imread(str(img_path))
            img_640  = cv2.resize(img_bgr, (640, 640))
            img_rgb  = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            t0 = time.time()
            results = yolo(str(img_path), device="cpu", imgsz=640, verbose=False)[0]
            t_total += time.time() - t0

            # Save plain detection
            det_img = draw_boxes(img_640, results, yolo.names)
            cv2.imwrite(str(det_dir / img_path.name), det_img)

            # Save Grad-CAM
            if len(results.boxes) > 0:
                run_gradcam(yolo, img_rgb, results, gcam_dir / f"gradcam_{img_path.name}")
                print(f"{len(results.boxes)} det ✓")
            else:
                print("no detections")

        avg_ms = t_total / len(image_files) * 1000
        summary[tag] = {"avg_ms": round(avg_ms, 1), "images": len(image_files)}
        print(f"\n  ⚡ {tag}: avg {avg_ms:.1f} ms/img on CPU")

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    for tag, info in summary.items():
        print(f"  {tag:10s} → {info['avg_ms']} ms/img | {info['images']} images")
    print(f"\n  Detection outputs  → {OUTPUT_DIRS['baseline']}/ and {OUTPUT_DIRS['cbam']}/")
    print(f"  Grad-CAM heatmaps  → {OUTPUT_DIRS['heatmap_baseline']}/ and {OUTPUT_DIRS['heatmap_cbam']}/")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "test_images"
    process_folder(folder)

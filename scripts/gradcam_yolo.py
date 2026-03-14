
from ultralytics import YOLO
import torch, cv2, numpy as np, os
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pathlib import Path

# ── Wrapper so GradCAM sees a single tensor output ──────────────────────────
class YOLOWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Return class scores only
        return out[:, 4:, :].mean(-1)  # [1, nc]

# ── Main ─────────────────────────────────────────────────────────────────────
def visualize_cam_folder(weights_path, images_folder, results_folder):
    yolo = YOLO(weights_path)
    yolo.to("cpu")  # force CPU

    os.makedirs(results_folder, exist_ok=True)

    # Collect images
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [p for p in Path(images_folder).glob("*") if p.suffix.lower() in image_extensions]
    if len(image_files) == 0:
        print("No images found in folder:", images_folder)
        return

    print(f"Found {len(image_files)} images. Starting Grad-CAM...")

    for img_path in image_files:
        print(f"Processing: {img_path.name}")

        # Detect
        results = yolo(str(img_path), device="cpu", imgsz=640, verbose=False)[0]
        if len(results.boxes) == 0:
            print("  No detections"); continue

        # Prepare image tensor for Grad-CAM
        img_bgr = cv2.imread(str(img_path))
        h_orig, w_orig = img_bgr.shape[:2]
        img_640 = cv2.resize(img_bgr, (640, 640))
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)

        # Grad-CAM
        wrapped = YOLOWrapper(yolo.model)
        target_layer = yolo.model.model[9]  # SPPF
        cam = EigenCAM(model=wrapped, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=tensor)[0, :]
        cam_img = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        # Draw bounding boxes for ALL detections
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf_val = box.conf[0].item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 = int(x1 * w_orig / 640)
            y1 = int(y1 * h_orig / 640)
            x2 = int(x2 * w_orig / 640)
            y2 = int(y2 * h_orig / 640)
            cv2.rectangle(cam_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(cam_img, f"{yolo.names[cls_id]} {conf_val:.2f}",
                        (x1, max(y1-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # Save Grad-CAM heatmap
        out_path = Path(results_folder) / f"gradcam_{img_path.name}"
        cv2.imwrite(str(out_path), cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))
        print(f"  Saved: {out_path.name}")

    print("Grad-CAM processing completed.")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    weights_path = input("Enter path to YOLOv8 model (.pt file): ").strip()
    images_folder = input("Enter path to folder containing images: ").strip()
    results_folder = input("Enter path to save Grad-CAM results: ").strip()

    visualize_cam_folder(weights_path, images_folder, results_folder)
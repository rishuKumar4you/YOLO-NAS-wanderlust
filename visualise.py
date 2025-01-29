import argparse
import yaml
import os
import cv2
import torch
import random
import numpy as np
import uuid
import json
import warnings
from super_gradients.training import models

# Suppress specific FutureWarnings from super_gradients
warnings.filterwarnings("ignore", category=FutureWarning, module="super_gradients")

import cv2
import random

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plots one bounding box on image img.
    """
    # Force a very thin line thickness
    tl = 1
    
    # Force a light green color in BGR
    # For example, a pastel-like green could be (144, 238, 144)
    color = (144, 238, 144)

    # Top-left corner (c1) and bottom-right corner (c2)
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # Draw the rectangle
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # Remove or comment out the label part to ensure no text is drawn
    # if label:
    #     tf = max(tl - 1, 1)
    #     t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
    #     c2_label = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
    #     cv2.rectangle(img, c1, c2_label, color, -1, cv2.LINE_AA)
    #     cv2.putText(img, label, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
    #                 tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

## commented out to remove the label, and forced the line thickness.

def main():
    # Argument Parsing
    ap = argparse.ArgumentParser(description="YOLO-NAS Image Inference Script")
    ap.add_argument("--model", type=str, required=True,
                    help="Model type (e.g., yolo_nas_s)")
    ap.add_argument("--image", type=str, required=True,
                    help="Path to the input image")
    ap.add_argument("--weights", type=str, default=None,
                    help="Path to the trained model weights (default: COCO weights)")
    ap.add_argument("--data", type=str, required=True,
                    help="Path to data.yaml file")
    ap.add_argument("--save", action='store_true',
                    help="Flag to save the annotated image and JSON output")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold for predictions (0 < conf < 1)")
    args = ap.parse_args()

    if not (0.0 < args.conf < 1.0):
        raise ValueError("Confidence threshold (--conf) must be between 0 and 1.")

    # Load data.yaml
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"data.yaml file not found at {args.data}")
    with open(args.data, 'r') as f:
        yaml_params = yaml.safe_load(f)

    class_names = yaml_params.get('names', [])
    if not class_names:
        raise ValueError("No class names found in data.yaml")

    random.seed(42)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

    # Load YOLO-NAS Model
    model = models.get(
        args.model,
        num_classes=len(class_names),
        checkpoint_path=args.weights if args.weights else None
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f'Loaded Model: {args.model}')
    print(f'Class Names: {class_names}')

    # Read Image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image file not found at {args.image}")
    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Failed to load image '{args.image}'")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform Inference
    with torch.no_grad():
        predictions = model.predict([img_rgb], conf=args.conf)

    predictions_json = {"predictions": []}
    if not predictions:
        print("[INFO] No predictions were made for the input image.")
    else:

        # Accessing predictions
        pred = predictions.prediction  # Directly access the 'prediction' attribute
        bboxes = np.array(pred.bboxes_xyxy)
        confs = pred.confidence
        labels = pred.labels.astype(int)

        for box, cnf, cls in zip(bboxes, confs, labels):
            class_name = predictions.class_names[cls]  # Access class name from 'class_names'
            color = colors[cls]
            plot_one_box(box[:4], img, label=f'{class_name} {cnf:.3f}', color=color)

            # Calculate x, y, width, height
            x, y, x2, y2 = box[:4]
            width = x2 - x
            height = y2 - y
            detection_id = str(uuid.uuid4())

            predictions_json["predictions"].append({
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "width": round(float(width), 2),
                "height": round(float(height), 2),
                "confidence": round(float(cnf), 3),
                "class": class_name,
                "class_id": int(cls),
                "detection_id": detection_id
            })
    # Save Output if required
    if args.save:
        filename = os.path.splitext(os.path.basename(args.image))[0]
        output_dir = os.path.join('outputs', filename)
        os.makedirs(output_dir, exist_ok=True)

        # Save annotated image
        annotated_image_path = os.path.join(output_dir, f"{filename}_labelled.jpg")
        cv2.imwrite(annotated_image_path, img)
        print(f"[INFO] Annotated image saved to {annotated_image_path}")

        # Save JSON
        json_path = os.path.join(output_dir, f"{filename}_labelled.json")
        with open(json_path, 'w') as jf:
            json.dump(predictions_json, jf, indent=4)
        print(f"[INFO] Predictions JSON saved to {json_path}")

    if not args.save:
        print(json.dumps(predictions_json, indent=4))

if __name__ == '__main__':
    main()

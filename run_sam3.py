"""
SAM3 video segmentation — per-frame open-vocabulary mask annotation.

Usage:
    python run_sam3.py <video_path> [options]

Example:
    python run_sam3.py ../data/videos/djurgarden1.mp4 --model sam3.pt
    python run_sam3.py ../data/videos/djurgarden1.mp4 --model sam3.pt --save-detections
"""

import argparse
import json
import os
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAM3 open-vocabulary segmentation on a video."
    )
    parser.add_argument("video_path", type=str, help="Input video file")
    parser.add_argument("--model", type=str, default="sam3.pt", help="SAM3 checkpoint path")
    parser.add_argument(
        "--text",
        nargs="+",
        default=[
            "basketball player wearing light blue jersey",
            "basketball player wearing black jersey",
            "referee",
            "ball",
        ],
        help="Text class prompts for open-vocabulary detection",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--half", action="store_true", help="FP16 inference (MPS/CUDA)")
    parser.add_argument("--device", type=str, default="", help="Device: '' (auto), '0', 'cuda:0', 'mps', 'cpu'")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory for visualizations (overrides SAM3_OUTPUT_DIR env var)")
    parser.add_argument("--save-detections", action="store_true", help="Save per-frame raw detections to output/detections/<stem>_sam3.json")
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics.models.sam import SAM3VideoSemanticPredictor

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    _default_out = Path(__file__).parent / "output" / "visualizations"
    out_dir = Path(args.output_dir) if args.output_dir else Path(os.environ.get("SAM3_OUTPUT_DIR", _default_out))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_sam3.mp4"

    output_root = out_dir.parent if out_dir.name == "visualizations" else out_dir
    dets_dir = output_root / "detections"
    if args.save_detections:
        dets_dir.mkdir(parents=True, exist_ok=True)
    dets_path = dets_dir / f"{video_path.stem}_sam3.json"

    # Read video metadata for detection JSON meta block
    _cap = cv2.VideoCapture(str(video_path))
    _fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    _total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    _cap.release()

    print(f"Input:  {video_path}")
    print(f"Model:  {args.model}")
    print(f"Prompts: {args.text}")
    print(f"Output: {out_path}")
    if args.save_detections:
        print(f"Detections: {dets_path}")
    print()

    overrides = dict(
        conf=args.conf,
        task="segment",
        mode="predict",
        imgsz=args.imgsz,
        model=args.model,
        half=args.half,
        device=args.device,
    )
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)

    results = predictor(
        source=str(video_path),
        text=args.text,
        stream=True,
    )

    writer = None
    frame_idx = 0
    t_start = time.time()
    frames_data = {}

    for result in results:
        annotated = result.plot(labels=False, conf=False)

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, 30.0, (w, h))

        writer.write(annotated)

        if args.save_detections:
            frame_records = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i])
                    frame_records.append({
                        "bbox": [round(v, 2) for v in boxes.xyxy[i].tolist()],
                        "conf": round(float(boxes.conf[i]), 4),
                        "class_id": class_id,
                        "class_name": args.text[class_id] if class_id < len(args.text) else str(class_id),
                    })
            frames_data[str(frame_idx)] = frame_records

        frame_idx += 1
        if frame_idx % 25 == 0:
            elapsed = time.time() - t_start
            fps = frame_idx / elapsed if elapsed > 0 else 0.0
            ids_seen = set()
            if result.boxes is not None and result.boxes.id is not None:
                ids_seen = set(result.boxes.id.int().tolist())
            print(f"Frame {frame_idx:5d} | {fps:5.1f} fps | IDs this frame: {sorted(ids_seen)}")

    if writer is not None:
        writer.release()
        print(f"\nDone — {frame_idx} frames written to {out_path}")
    else:
        print("No frames processed.")

    if args.save_detections:
        det_output = {
            "meta": {
                "model": args.model,
                "source": "sam3",
                "imgsz": args.imgsz,
                "fps": _fps,
                "total_frames": frame_idx,
                "classes": args.text,
            },
            "frames": frames_data,
        }
        with open(dets_path, "w") as f:
            json.dump(det_output, f)
        print(f"Detections written to {dets_path}")


if __name__ == "__main__":
    main()

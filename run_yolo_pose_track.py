"""
YOLO11x-pose detection + BoT-SORT tracking on a video.

Single pass gives: detection boxes, stable track IDs, and 17 COCO keypoints per person.

Usage:
    python run_yolo_pose_track.py <video_path> [options]

Example:
    python run_yolo_pose_track.py ../data/videos/djurgarden1.mp4 --device mps --half
"""

import argparse
import json
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO11x-pose + BoT-SORT tracking on a video."
    )
    parser.add_argument("video_path", type=str, help="Input video file")
    parser.add_argument(
        "--model", type=str, default="yolo11x-pose.pt", help="YOLO pose model (auto-downloads)"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker: botsort (appearance+Kalman) or bytetrack (IoU-only, faster)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")
    parser.add_argument("--half", action="store_true", help="FP16 inference (MPS/CUDA)")
    parser.add_argument(
        "--device", type=str, default="", help="Device: '' (auto), '0', 'mps', 'cpu'"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Root output directory"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from ultralytics import YOLO

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "visualizations"
    tracks_dir = output_dir / "tracks"
    viz_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    out_video = viz_dir / f"{video_path.stem}_yolo_pose.mp4"
    out_json = tracks_dir / f"{video_path.stem}_yolo_pose.json"

    print(f"Input:   {video_path}")
    print(f"Model:   {args.model}")
    print(f"Tracker: {args.tracker}")
    print(f"Output video: {out_video}")
    print(f"Output JSON:  {out_json}")
    print()

    model = YOLO(args.model)

    # Read video metadata for JSON meta block
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    results_iter = model.track(
        source=str(video_path),
        tracker=f"{args.tracker}.yaml",
        conf=args.conf,
        imgsz=args.imgsz,
        half=args.half,
        device=args.device,
        stream=True,
        verbose=False,
    )

    writer = None
    frame_idx = 0
    t_start = time.time()
    frames_data = {}

    for result in results_iter:
        annotated = result.plot(labels=False, conf=False)

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

        writer.write(annotated)

        # Build per-frame JSON data
        frame_records = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            kpts = result.keypoints  # may be None if no detections

            for i in range(len(boxes)):
                track_id = int(boxes.id[i]) if boxes.id is not None else -1
                xyxy = boxes.xyxy[i].tolist()
                conf = float(boxes.conf[i])

                keypoints = []
                if kpts is not None and i < len(kpts):
                    kp_data = kpts.data[i]  # shape (17, 3): x, y, conf
                    keypoints = kp_data.tolist()

                frame_records.append(
                    {
                        "id": track_id,
                        "bbox": [round(v, 2) for v in xyxy],
                        "conf": round(conf, 4),
                        "keypoints": [[round(x, 2), round(y, 2), round(c, 4)] for x, y, c in keypoints],
                    }
                )

        frames_data[str(frame_idx)] = frame_records

        frame_idx += 1
        if frame_idx % 25 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            active_ids = [r["id"] for r in frame_records]
            print(f"Frame {frame_idx:5d} | {fps_proc:5.1f} fps | IDs this frame: {sorted(active_ids)}")

    if writer is not None:
        writer.release()
        print(f"\nDone — {frame_idx} frames written to {out_video}")
    else:
        print("No frames processed.")

    track_output = {
        "meta": {
            "model": args.model,
            "tracker": args.tracker,
            "imgsz": args.imgsz,
            "fps": fps,
            "total_frames": frame_idx,
        },
        "frames": frames_data,
    }
    with open(out_json, "w") as f:
        json.dump(track_output, f)
    print(f"Tracks written to {out_json}")


if __name__ == "__main__":
    main()

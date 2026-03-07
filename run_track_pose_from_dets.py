"""
Stage-2 pipeline: load a detection checkpoint JSON + original video
→ apply external BoT-SORT/ByteTrack tracking + YOLO11x-pose keypoints
→ save tracks JSON (same schema as run_yolo_pose_track.py) + viz video.

Usage:
    python run_track_pose_from_dets.py <detections_json> <video_path> [options]

Example:
    python run_track_pose_from_dets.py \\
        output/detections/djurgarden1_sam3.json \\
        ../data/videos/djurgarden1.mp4 \\
        --device mps --half
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Track + pose from a detection checkpoint JSON."
    )
    parser.add_argument("detections_json", type=str, help="Path to detection checkpoint JSON")
    parser.add_argument("video_path", type=str, help="Original video file")
    parser.add_argument("--pose-model", type=str, default="yolo11x-pose.pt", help="YOLO pose model")
    parser.add_argument(
        "--tracker",
        type=str,
        default="botsort",
        choices=["botsort", "bytetrack"],
        help="Tracker: botsort or bytetrack",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Min confidence threshold")
    parser.add_argument("--device", type=str, default="", help="Device: '' (auto), 'mps', 'cpu', '0'")
    parser.add_argument("--half", action="store_true", help="FP16 for pose model")
    parser.add_argument("--output-dir", type=str, default="output", help="Root output directory")
    return parser.parse_args()


def compute_iou(box_a, box_b):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def main():
    args = parse_args()

    from ultralytics import YOLO
    from boxmot import BotSort, ByteTracker

    dets_json_path = Path(args.detections_json)
    video_path = Path(args.video_path)

    if not dets_json_path.exists():
        raise FileNotFoundError(f"Detections JSON not found: {dets_json_path}")
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(args.output_dir)
    viz_dir = output_dir / "visualizations"
    tracks_dir = output_dir / "tracks"
    viz_dir.mkdir(parents=True, exist_ok=True)
    tracks_dir.mkdir(parents=True, exist_ok=True)

    # Derive stem from detections JSON name (e.g. djurgarden1_sam3 → djurgarden1_sam3_tracked)
    det_stem = dets_json_path.stem  # e.g. "djurgarden1_sam3"
    out_video = viz_dir / f"{det_stem}_tracked.mp4"
    out_json = tracks_dir / f"{det_stem}_tracked.json"

    with open(dets_json_path) as f:
        dets = json.load(f)

    meta = dets["meta"]
    fps = meta.get("fps", 30.0)
    total_frames = meta.get("total_frames", 0)

    print(f"Detections: {dets_json_path}")
    print(f"Video:      {video_path}")
    print(f"Pose model: {args.pose_model}")
    print(f"Tracker:    {args.tracker}")
    print(f"Output video: {out_video}")
    print(f"Output JSON:  {out_json}")
    print()

    # Init tracker
    if args.tracker == "botsort":
        tracker = BotSort(reid_weights=Path("osnet_x0_25_msmt17.pt"), device=args.device or "cpu", half=args.half)
    else:
        tracker = ByteTracker()

    # Init pose model
    pose_model = YOLO(args.pose_model)

    cap = cv2.VideoCapture(str(video_path))
    writer = None
    frame_idx = 0
    t_start = time.time()
    frames_data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_dets = dets["frames"].get(str(frame_idx), [])

        # Filter by confidence threshold
        frame_dets = [d for d in frame_dets if d["conf"] >= args.conf]

        if frame_dets:
            det_array = np.array(
                [[*d["bbox"], d["conf"], d["class_id"]] for d in frame_dets],
                dtype=np.float32,
            )
        else:
            det_array = np.empty((0, 6), dtype=np.float32)

        # Update tracker → returns (M, 8): x1,y1,x2,y2,track_id,conf,cls,idx
        tracks = tracker.update(det_array, frame)

        # Run pose on full frame
        pose_results = pose_model(frame, verbose=False, half=args.half, device=args.device)

        # Collect pose detections: list of (bbox, keypoints)
        pose_boxes = []
        pose_kpts = []
        if pose_results and pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
            pb = pose_results[0].boxes
            pk = pose_results[0].keypoints
            for i in range(len(pb)):
                pose_boxes.append(pb.xyxy[i].tolist())
                if pk is not None and i < len(pk):
                    pose_kpts.append(pk.data[i].tolist())
                else:
                    pose_kpts.append([])

        # Build per-frame records
        frame_records = []
        if tracks is not None and len(tracks) > 0:
            for track in tracks:
                x1, y1, x2, y2, track_id, conf, cls, _ = track
                track_bbox = [float(x1), float(y1), float(x2), float(y2)]

                # Match to best pose detection by IoU
                keypoints = []
                best_iou = 0.0
                for pb, pk in zip(pose_boxes, pose_kpts):
                    iou = compute_iou(track_bbox, pb)
                    if iou > best_iou:
                        best_iou = iou
                        if iou > 0.4:
                            keypoints = pk

                frame_records.append({
                    "id": int(track_id),
                    "bbox": [round(v, 2) for v in track_bbox],
                    "conf": round(float(conf), 4),
                    "keypoints": [[round(x, 2), round(y, 2), round(c, 4)] for x, y, c in keypoints],
                })

        frames_data[str(frame_idx)] = frame_records

        # Annotate frame with tracks and skeletons
        annotated = frame.copy()
        for record in frame_records:
            bx1, by1, bx2, by2 = [int(v) for v in record["bbox"]]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            cv2.putText(annotated, str(record["id"]), (bx1, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for kp in record["keypoints"]:
                kx, ky, kc = kp
                if kc > 0.3:
                    cv2.circle(annotated, (int(kx), int(ky)), 3, (0, 0, 255), -1)

        if writer is None:
            h, w = annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 25 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
            active_ids = [r["id"] for r in frame_records]
            print(f"Frame {frame_idx:5d} | {fps_proc:5.1f} fps | IDs this frame: {sorted(active_ids)}")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\nDone — {frame_idx} frames written to {out_video}")
    else:
        print("No frames processed.")

    track_output = {
        "meta": {
            "model": args.pose_model,
            "tracker": args.tracker,
            "source": meta.get("source", "unknown"),
            "detections_json": str(dets_json_path),
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

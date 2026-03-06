# SAM3 Experiments — Notes

## Speed optimization tips

### Quick wins (no architecture changes)
- `--imgsz 320` — quarters compute vs 640; big speed gain at cost of quality for distant players
- Drop `"ball"` from prompts — severe over-detection, reduces per-frame work
- `--conf 0.4–0.5` — fewer detections = faster postprocess

### Architecture changes
- **Frame subsampling** — process every 3rd frame (10fps instead of 30fps); sufficient for SGA-Interact graph input
- **Two-stage pipeline** — YOLO for fast bounding boxes → SAM3 only on player crops (~128×128 instead of full frame); ~25× less SAM3 compute per frame
- **MPS on local** — `--device mps` on M4 Pro; expect 3–8× over CPU

### For full games
- **Colab A100** at `--imgsz 1280 --half --device 0`: expect 5–15 fps → 1.5h game in ~3–9h as overnight batch job

---

## Sports-focused tracking libraries

These are specifically designed for team sports and handle occlusions + re-ID better than ByteTrack/BotSORT:

- **StrongSORT + OSNet** — much better appearance-based re-ID than ByteTrack
- **OC-SORT** — built for occlusion handling
- **BoT-SORT** — with a fine-tuned re-ID head on jersey colors/numbers
- **SportsMOT** — dataset + models trained on basketball/soccer multi-object tracking benchmarks
- **DanceTrack** models — handle crowded, similar-appearance targets
- **SoccerNet tracking challenge** solutions — state-of-the-art for team sports tracking

---

## Detection quality observations (djurgarden1.mp4, CPU run)

- Player counts consistent: ~6–7 light blue, ~5–6 black per frame ✓
- Referees over-counted: 3–4 detected vs ~2 real — misclassification at conf=0.25
- Ball: severe false positives (10–19/frame) — "ball" prompt too loose, drop it
- IDs climbing into 100s by frame 300 — tracker not maintaining consistent identities

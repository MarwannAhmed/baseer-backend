import cv2
import csv
import re
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

KNOWN_COLORS = {
    'red', 'green', 'blue', 'yellow', 'orange',
    'brown', 'purple', 'pink', 'white', 'gray', 'black'
}

VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.3gp', '.wmv'}

# Resize every saved crop to this — standard input size for classifiers
CROP_SIZE = 224

def parse_label(filename):

    stem = Path(filename).stem.lower()
    return re.sub(r'_\d+$', '', stem)  


def extract_frames(video_path: Path, n: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open {video_path.name}")
        return []

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps

    if total < 1:
        cap.release()
        return []

    n = min(n, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    print(f"  {len(frames)} frames  |  {total} total frames  |  {duration:.1f}s @ {fps:.0f}fps")
    return frames


def select_roi(first_frame, label):

    orig_height, orig_width = first_frame.shape[:2]

    max_side = 900
    scale    = min(max_side / orig_width, max_side / orig_height, 1.0)
    disp_w   = int(orig_width * scale)
    disp_h   = int(orig_height * scale)
    display  = cv2.resize(first_frame, (disp_w, disp_h)) if scale < 1.0 else first_frame.copy()

    banner_h = 45
    banner = np.zeros((banner_h, disp_w, 3), dtype=np.uint8)
    msg = f"[{label}]  Draw box around object  |  SPACE/ENTER = confirm  |  C = full frame"
    cv2.putText(banner, msg, (8, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    display_with_banner = np.vstack([banner, display])

    roi = cv2.selectROI(f"ROI selection — {label}", display_with_banner,
                        fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    rx, ry, rw, rh = roi
    ry -= banner_h        

    if rw < 5 or rh < 5:
        print("No ROI drawn → using full frame")
        return (0, 0, orig_width, orig_height)

    if scale < 1.0:
        rx = int(rx / scale);  ry = int(ry / scale)
        rw = int(rw / scale);  rh = int(rh / scale)

    rx = max(0, min(rx, orig_width - 1))
    ry = max(0, min(ry, orig_height - 1))
    rw = min(rw, orig_width - rx)
    rh = min(rh, orig_height - ry)

    return (rx, ry, rw, rh)


def build_dataset(videos_dir, output_dir, n_frames):
    videos_dir = Path(videos_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted([
        f for f in videos_dir.iterdir()
        if f.suffix.lower() in VIDEO_EXTENSIONS
    ])

    if not video_files:
        print(f"[Error] No video files found in: {videos_dir}")
        sys.exit(1)

    print(f"\nFound {len(video_files)} video(s):\n")
    for f in video_files:
        lbl  = parse_label(f.name)
        warn = "" if lbl in KNOWN_COLORS else "  ⚠  not in the 11 known colors"
        print(f"  {f.name:<35} →  label: '{lbl}'{warn}")

    unknown = [parse_label(f.name) for f in video_files
               if parse_label(f.name) not in KNOWN_COLORS]
    if unknown:
        print(f"\n  ⚠  Labels not matching the app's 11 colors: {set(unknown)}")
        print(f"     Known colors: {sorted(KNOWN_COLORS)}")
        print(f"     Rename the video files if this is a mistake, or continue anyway.")
        ans = input("\n  Continue? [y/N] ").strip().lower()
        if ans != 'y':
            sys.exit(0)

    print()

    records      = []                   
    label_counts = defaultdict(int)    

    for video_path in video_files:
        label = parse_label(video_path.name)
        print(f"── {video_path.name}  ({label}) " + "─" * 40)

        frames = extract_frames(video_path, n_frames)
        if not frames:
            print("  Skipping — could not read any frames\n")
            continue

        roi = select_roi(frames[0], label)
        x, y, w, h = roi
        print(f"  ROI: x={x} y={y}  w={w} h={h}")

        label_dir = output_dir / label
        label_dir.mkdir(exist_ok=True)

        saved = 0
        for frame in frames:
            crop = frame[y : y + h, x : x + w]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)

            filename = f"{label}_{label_counts[label]:04d}.jpg"
            out_path = label_dir / filename
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

            records.append((str(out_path.relative_to(output_dir)), label))
            label_counts[label] += 1
            saved += 1

        print(f"  Saved {saved} crops → {label_dir.name}/\n")

    csv_path = output_dir / "labels.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'label'])
        writer.writerows(records)

    print("=" * 50)
    print(f"Total:  {len(records)} samples  |  {len(label_counts)} classes")
    print(f"Folder: {output_dir.resolve()}")
    print(f"CSV:    {csv_path.resolve()}\n")

    max_count = max(label_counts.values(), default=1)
    print("Samples per class:")
    for lbl, cnt in sorted(label_counts.items()):
        bar   = "█" * int(cnt / max_count * 30)
        check = "✓" if lbl in KNOWN_COLORS else "?"
        print(f"  {check} {lbl:<14}  {cnt:>3}  {bar}")

    missing = KNOWN_COLORS - set(label_counts.keys())
    if missing:
        print(f"\n  Missing classes: {sorted(missing)}")

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build a color dataset from short demo videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_dataset.py --videos ./videos
  python build_dataset.py --videos ./videos --output ./dataset --frames 20

Naming convention:
  red.mp4, blue.mp4, green.mp4       one video per color
  red_1.mp4, red_2.mp4               multiple clips → same label "red"
        """
    )
    ap.add_argument('--videos',  required=True,
                    help='Folder containing the video files')
    ap.add_argument('--output',  default='dataset',
                    help='Output folder (default: ./dataset)')
    ap.add_argument('--frames',  default=15, type=int,
                    help='Frames to sample per video (default: 15)')
    args = ap.parse_args()

    build_dataset(args.videos, args.output, args.frames)
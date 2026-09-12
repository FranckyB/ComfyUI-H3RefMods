"""Generate random PNG thumbnails for MP4 files in a directory.

Each thumbnail is extracted at full frame resolution from a random point
between 25% and 75% of the video's duration, then saved beside the source MP4
with the same basename and a `.png` extension.

Rerun the script to regenerate different thumbnails.

Usage:
  python3 generate_video_thumbnails.py /path/to/folder
  python3 generate_video_thumbnails.py /path/to/folder --seed 1234
"""

from __future__ import annotations

import argparse
import random
import subprocess  # nosec B404
import sys
from pathlib import Path


def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec B603
    value = (result.stdout or "").strip()
    if not value:
        raise ValueError(f"Could not read duration for {path}")
    duration = float(value)
    if duration <= 0.0:
        raise ValueError(f"Invalid duration for {path}: {duration}")
    return duration


def _pick_timestamp(duration: float, rng: random.Random) -> float:
    if duration <= 0.0:
        return 0.0
    start = duration * 0.25
    end = duration * 0.75
    if end <= start:
        return duration * 0.5
    return rng.uniform(start, end)


def _render_thumbnail(video_path: Path, timestamp: float) -> Path:
    output_path = video_path.with_suffix(".png")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-update",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # nosec B603
    return output_path


def _iter_mp4s(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    )


def _resolve_folder(folder_arg: str | None) -> Path:
    folder_text = folder_arg
    if not folder_text:
        try:
            folder_text = input("Folder containing MP4 files: ").strip()
        except EOFError:
            folder_text = ""
    if not folder_text:
        raise ValueError("No folder provided.")
    return Path(folder_text).expanduser().resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a random full-resolution PNG thumbnail for every MP4 in a folder. "
            "The capture time is chosen between 25% and 75% of each video's duration."
        )
    )
    parser.add_argument("folder", nargs="?", help="Folder containing MP4 files.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable thumbnail choices.",
    )
    args = parser.parse_args()

    try:
        folder = _resolve_folder(args.folder)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    videos = _iter_mp4s(folder)
    if not videos:
        print(f"No MP4 files found in {folder}", file=sys.stderr)
        return 1

    failures = 0
    for video_path in videos:
        try:
            duration = _probe_duration(video_path)
            timestamp = _pick_timestamp(duration, rng)
            output_path = _render_thumbnail(video_path, timestamp)
            print(f"{video_path.name} -> {output_path.name} at {timestamp:.2f}s / {duration:.2f}s")
        except Exception as exc:
            failures += 1
            print(f"Failed: {video_path.name}: {exc}", file=sys.stderr)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
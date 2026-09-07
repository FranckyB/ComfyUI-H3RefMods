"""Generate MiniMax H3 RefMod (.safetensors) from a dataset folder.

Automatically scans the dataset directory for reference images, videos, and
audio clips, invokes ComfyUI's extract_mod.py pipeline with optimal identity
settings, and saves/copies the resulting RefMod to both ComfyUI's model tree
and the dataset folder.

Our workflow (Linux):
  - ComfyUI models:  /mnt/Neuralnet/ComfyUI/models/
  - Datasets:        /mnt/Neuralnet/Datasets/

Usage:
  python3 generate_refmod.py <folder_or_dataset_name> [options]

Examples:
  # Generate RefMod for a dataset folder (auto-discovers images & video clips)
  python3 generate_refmod.py /mnt/Neuralnet/Musubi-Trainer/Datasets/Sleepy

  # Generate by dataset name under the datasets root
  python3 generate_refmod.py Sleepy

  # Preview media discovered and execution command without encoding
  python3 generate_refmod.py Sleepy --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

# Ensure UTF-8 output on Windows consoles
reconfig_out = getattr(sys.stdout, "reconfigure", None)
if callable(reconfig_out):
    reconfig_out(encoding="utf-8", errors="replace")
reconfig_err = getattr(sys.stderr, "reconfigure", None)
if callable(reconfig_err):
    reconfig_err(encoding="utf-8", errors="replace")

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
    ".m4v",
    ".flv",
    ".wmv",
    ".ts",
}

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".aac",
    ".m4a",
    ".ogg",
    ".opus",
}

DEFAULT_DATASETS_ROOTS = [
    Path("/mnt/Neuralnet/Tools/H3_Mods/"),
    Path("datasets"),
]

DEFAULT_COMFYUI_ROOTS = [
    Path("/mnt/Neuralnet/ComfyUI"),
]


class DatasetMedia(NamedTuple):
    images: list[Path]
    videos: list[Path]
    audios: list[Path]


class ComfyUIEnvironment(NamedTuple):
    comfy_dir: Path
    python_exe: Path
    extract_mod_script: Path
    vae_path: Path
    refmods_dir: Path
    audio_vae_path: Path | None = None


class RefModResult(NamedTuple):
    name: str
    comfy_output_path: Path
    dataset_output_path: Path | None
    images_count: int
    videos_count: int
    audios_count: int
    token_count: int | None
    size_mb: float | None


def sanitize_stem(stem: str) -> str:
    """Create a clean, filesystem-safe identifier from a folder or file name."""
    normalized = unicodedata.normalize("NFKD", stem)
    cleaned = re.sub(r"\[[^\]]*\]", "", normalized)
    cleaned = re.sub(r"\([^\)]*\)", "", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned if cleaned else "concept"


def resolve_dataset_folder(folder_input: str | Path) -> Path:
    """Resolve dataset folder from absolute, relative, or known dataset roots."""
    path = Path(folder_input).resolve()
    if path.exists() and path.is_dir():
        return path

    folder_str = str(folder_input).strip("\"'")
    direct_path = Path(folder_str)
    if direct_path.exists() and direct_path.is_dir():
        return direct_path.resolve()

    # If it's a short name, search standard dataset directories
    for root in DEFAULT_DATASETS_ROOTS:
        candidate = root / folder_str
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()

    raise ValueError(f"Dataset directory not found: '{folder_input}'")


def _first_existing(candidates: Sequence[Path]) -> Path | None:
    """Return the first candidate path that exists, resolved, or None."""
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_comfy_dir(comfy_dir: Path | None) -> Path:
    if comfy_dir is not None and comfy_dir.exists():
        return comfy_dir

    env_comfy = os.environ.get("COMFYUI_PATH")
    if env_comfy and Path(env_comfy).exists():
        return Path(env_comfy).resolve()

    resolved = _first_existing([c for c in DEFAULT_COMFYUI_ROOTS if c.is_dir()])
    if resolved is not None:
        return resolved

    raise ValueError(
        "Could not locate ComfyUI installation directory. "
        "Please specify --comfy-dir or set COMFYUI_PATH environment variable."
    )


def _resolve_python_exe(comfy_dir: Path, python_exe: Path | None) -> Path:
    if python_exe is not None and python_exe.exists():
        return python_exe

    # NOTE: do NOT resolve() these — venv/bin/python is typically a symlink to
    # the system interpreter, and resolving it would drop the venv's
    # site-packages (torch, safetensors, ...). Return the venv path as-is so
    # the venv is actually used.
    candidates = [
        comfy_dir / "venv" / "bin" / "python",
        comfy_dir / "venv" / "Scripts" / "python.exe",
        comfy_dir / "python_embeded" / "python.exe",
        comfy_dir.parent / "python_embeded" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _resolve_extract_mod_script(comfy_dir: Path, extract_mod_script: Path | None) -> Path:
    if extract_mod_script is not None and extract_mod_script.exists():
        return extract_mod_script

    candidates = [
        comfy_dir / "custom_nodes" / "ComfyUI-H3RefMods" / "tools" / "extract_mod.py",
        Path("custom_nodes/ComfyUI-H3RefMods/tools/extract_mod.py"),
    ]
    resolved = _first_existing(candidates)
    if resolved is None:
        raise ValueError(
            f"Could not locate extract_mod.py under ComfyUI custom_nodes: {comfy_dir}. "
            "Ensure ComfyUI-H3RefMods is installed."
        )
    return resolved


def _resolve_vae_path(comfy_dir: Path, vae_path: Path | None) -> Path:
    if vae_path is not None and vae_path.exists():
        return vae_path

    candidates = [
        comfy_dir / "models" / "vae" / "MiniMax" / "minimax_h3_video_vae_fp16.safetensors",
        comfy_dir / "models" / "vae" / "MiniMax" / "minimax_h3_video_vae_int8_convrot.safetensors"
    ]
    resolved = _first_existing(candidates)
    if resolved is None:
        raise ValueError(
            f"Could not locate MiniMax H3 Video VAE in {comfy_dir / 'models' / 'vae'}. "
            "Please provide --vae path explicitly."
        )
    return resolved


def _resolve_audio_vae_path(comfy_dir: Path, audio_vae_path: Path | None) -> Path | None:
    """Resolve the MiniMax H3 audio VAE; returns None if not found (audio optional)."""
    if audio_vae_path is not None and audio_vae_path.exists():
        return audio_vae_path
    candidates = [
        comfy_dir / "models" / "vae" / "MiniMax" / "minimax_h3_audio_vae_fp32.safetensors",
        comfy_dir / "models" / "vae" / "minimax_h3_audio_vae_fp32.safetensors",
    ]
    return _first_existing(candidates)


def discover_comfyui_environment(
    comfy_dir: Path | None = None,
    python_exe: Path | None = None,
    extract_mod_script: Path | None = None,
    vae_path: Path | None = None,
    refmods_dir: Path | None = None,
    audio_vae_path: Path | None = None,
) -> ComfyUIEnvironment:
    """Discover ComfyUI installation, Python virtualenv, extract_mod.py script, and H3 VAE."""
    resolved_comfy = _resolve_comfy_dir(comfy_dir)
    resolved_python = _resolve_python_exe(resolved_comfy, python_exe)
    resolved_script = _resolve_extract_mod_script(resolved_comfy, extract_mod_script)
    resolved_vae = _resolve_vae_path(resolved_comfy, vae_path)
    resolved_audio_vae = _resolve_audio_vae_path(resolved_comfy, audio_vae_path)

    resolved_refmods = refmods_dir or (resolved_comfy / "models" / "refmods").resolve()
    resolved_refmods.mkdir(parents=True, exist_ok=True)

    return ComfyUIEnvironment(
        comfy_dir=resolved_comfy,
        python_exe=resolved_python,
        extract_mod_script=resolved_script,
        vae_path=resolved_vae,
        refmods_dir=resolved_refmods,
        audio_vae_path=resolved_audio_vae,
    )


def natural_sort_key(s: str | Path) -> list[int | str]:
    """Sort strings containing numbers in human/natural order."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


@dataclass
class _MediaBuckets:
    images: list[Path] = field(default_factory=list)
    videos: list[Path] = field(default_factory=list)
    audios: list[Path] = field(default_factory=list)


def _classify_file(file_path: Path, buckets: _MediaBuckets) -> None:
    ext = file_path.suffix.lower()
    if ext in SUPPORTED_IMAGE_EXTENSIONS:
        buckets.images.append(file_path)
    elif ext in SUPPORTED_VIDEO_EXTENSIONS:
        buckets.videos.append(file_path)
    elif ext in SUPPORTED_AUDIO_EXTENSIONS:
        buckets.audios.append(file_path)


def _scan_recursive(folder: Path, buckets: _MediaBuckets) -> None:
    for entry in folder.rglob("*"):
        if entry.is_file():
            _classify_file(entry, buckets)


def _scan_top_level(folder: Path, buckets: _MediaBuckets) -> None:
    for entry in folder.iterdir():
        if entry.is_file():
            _classify_file(entry, buckets)


def _scan_subfolder_by_extension(subfolder: Path, extensions: set[str]) -> list[Path]:
    """Files directly under subfolder whose extension is in extensions, if it exists at all."""
    if not subfolder.is_dir():
        return []
    return [
        entry
        for entry in subfolder.iterdir()
        if entry.is_file() and entry.suffix.lower() in extensions
    ]


def _fill_missing_from_subfolders(folder: Path, buckets: _MediaBuckets) -> None:
    """A non-recursive scan misses clips/audio kept in dedicated subfolders — fill from those."""
    if not buckets.videos:
        buckets.videos.extend(
            _scan_subfolder_by_extension(folder / "clips", SUPPORTED_VIDEO_EXTENSIONS)
        )

    if not buckets.audios:
        for sub in ("audio", "clips_audio"):
            buckets.audios.extend(
                _scan_subfolder_by_extension(folder / sub, SUPPORTED_AUDIO_EXTENSIONS)
            )


def scan_dataset_media(folder: Path, recursive: bool = False) -> DatasetMedia:
    """Scan folder for reference images, video clips, and audio files."""
    buckets = _MediaBuckets()

    if recursive:
        _scan_recursive(folder, buckets)
    else:
        _scan_top_level(folder, buckets)
        _fill_missing_from_subfolders(folder, buckets)

    buckets.images.sort(key=natural_sort_key)
    buckets.videos.sort(key=natural_sort_key)
    buckets.audios.sort(key=natural_sort_key)

    return DatasetMedia(images=buckets.images, videos=buckets.videos, audios=buckets.audios)


def derive_refmod_name(folder: Path, explicit_name: str | None = None) -> str:
    """Derive standard RefMod identifier (e.g. minimaxh3_<persona>_v1_refmod)."""
    if explicit_name:
        clean = explicit_name.strip()
        if clean.lower().endswith(".safetensors"):
            clean = clean[:-12]
        return clean

    stem = sanitize_stem(folder.name)
    if stem.startswith("minimaxh3_") and stem.endswith("_refmod"):
        return stem
    return f"minimaxh3_{stem}_v1_refmod"


@dataclass
class ComfyUIEnvOverrides:
    """Explicit overrides for discover_comfyui_environment; a field left None auto-discovers."""

    comfy_dir: Path | None = None
    python_exe: Path | None = None
    extract_mod_script: Path | None = None
    vae_path: Path | None = None
    refmods_dir: Path | None = None


def generate_refmod(
    folder_input: str | Path,
    name: str | None = None,
    mode: str = "encode",
    concept_type: str = "identity",
    resolution: int = 1024,
    max_tokens: int = 8192,
    description: str | None = None,
    env_overrides: ComfyUIEnvOverrides | None = None,
    copy_to_dataset: bool = False,
    recursive: bool = False,
    dry_run: bool = False,
) -> RefModResult:
    """Scan dataset media and invoke extract_mod.py to generate RefMod."""
    overrides = env_overrides or ComfyUIEnvOverrides()
    folder = resolve_dataset_folder(folder_input)
    env = discover_comfyui_environment(
        comfy_dir=overrides.comfy_dir,
        python_exe=overrides.python_exe,
        extract_mod_script=overrides.extract_mod_script,
        vae_path=overrides.vae_path,
        refmods_dir=overrides.refmods_dir,
    )

    media = scan_dataset_media(folder, recursive=recursive)
    total_visual_refs = len(media.images) + len(media.videos)

    print("\n========================================================")
    print("  MiniMax H3 RefMod Generator")
    print("========================================================")
    print(f"Dataset Folder: {folder}")
    print("Found Media:")
    print(f"  - Images: {len(media.images)}")
    print(f"  - Videos: {len(media.videos)}")
    print(f"  - Audios: {len(media.audios)}")

    if total_visual_refs == 0:
        raise ValueError(
            f"No supported images or videos found in '{folder}'. "
            "MiniMax H3 RefMod extraction requires at least one image or video reference."
        )

    mod_name = derive_refmod_name(folder, name)
    mod_desc = description or f"{folder.name} persona reference mod"

    print("\nRefMod Configuration:")
    print(f"  Name:         {mod_name}")
    print(f"  Mode:         {mode}")
    print(f"  Concept Type: {concept_type}")
    print(f"  Resolution:   {resolution}px short-edge")
    print(f"  Max Tokens:   {max_tokens}")
    print(f"  Description:  {mod_desc}")
    print(f"  VAE:          {env.vae_path.name}")
    print(f"  Output Dir:   {env.refmods_dir}")

    cmd_args: list[str] = [
        str(env.extract_mod_script),
        "--vae",
        str(env.vae_path),
        "--name",
        mod_name,
        "--mode",
        mode,
        "--concept-type",
        concept_type,
        "--resolution",
        str(resolution),
        "--max-tokens",
        str(max_tokens),
        "--output",
        str(env.refmods_dir),
        "--description",
        mod_desc,
    ]

    for img in media.images:
        cmd_args.extend(["--image", str(img)])
    for vid in media.videos:
        cmd_args.extend(["--video", str(vid)])

    # Attach audio identity when the dataset has audio and the audio VAE is available.
    if media.audios:
        if env.audio_vae_path is not None:
            cmd_args.extend(["--audio-vae", str(env.audio_vae_path)])
            for aud in media.audios:
                cmd_args.extend(["--audio", str(aud)])
        else:
            print(
                f"  [note] {len(media.audios)} audio file(s) found but no MiniMax H3 "
                "audio VAE was located — audio will NOT be embedded in the RefMod."
            )

    full_cmd = [str(env.python_exe), *cmd_args]

    if dry_run:
        print(
            f"\n[DRY RUN] Would execute command with {len(media.images)} image(s) and {len(media.videos)} video(s):"
        )
        print(f"  {' '.join(full_cmd[:10])} ... [+{total_visual_refs} media paths]")
        return RefModResult(
            name=mod_name,
            comfy_output_path=env.refmods_dir / f"{mod_name}.safetensors",
            dataset_output_path=folder / f"{mod_name}.safetensors" if copy_to_dataset else None,
            images_count=len(media.images),
            videos_count=len(media.videos),
            audios_count=len(media.audios),
            token_count=None,
            size_mb=None,
        )

    print("\nExtracting RefMod latents with ComfyUI H3 VAE...")
    proc = subprocess.run(  # nosec B603
        full_cmd,
        check=True,
        text=True,
    )

    if proc.returncode != 0:
        raise RuntimeError(f"extract_mod.py failed with exit code {proc.returncode}")

    comfy_mod_file = env.refmods_dir / f"{mod_name}.safetensors"
    dataset_mod_file: Path | None = None

    if not comfy_mod_file.exists():
        raise RuntimeError(f"Expected output file was not created: {comfy_mod_file}")

    size_bytes = comfy_mod_file.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    if copy_to_dataset:
        dataset_mod_file = folder / f"{mod_name}.safetensors"
        shutil.copy2(comfy_mod_file, dataset_mod_file)
        print(f"\nCopied RefMod to dataset folder:\n  -> {dataset_mod_file}")

    print("\nRefMod generated successfully!")
    print(f"  File: {comfy_mod_file} ({size_mb:.2f} MB)")
    print(f"  Ready for use with 'Load H3 RefMods' (dropdown '{mod_name}')")

    return RefModResult(
        name=mod_name,
        comfy_output_path=comfy_mod_file,
        dataset_output_path=dataset_mod_file,
        images_count=len(media.images),
        videos_count=len(media.videos),
        audios_count=len(media.audios),
        token_count=None,
        size_mb=size_mb,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MiniMax H3 RefMod (.safetensors) from a dataset folder."
    )
    parser.add_argument(
        "folder",
        help="Path or name of the dataset folder containing reference images, videos, and audio.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output RefMod name (default: minimaxh3_<folder_stem>_v1_refmod).",
    )
    parser.add_argument(
        "--mode",
        choices=["encode", "training"],
        default="encode",
        help="RefMod mode: 'encode' = full VAE identity encode (recommended for persons), 'training' = pooled grid.",
    )
    parser.add_argument(
        "--concept-type",
        default="identity",
        choices=[
            "identity",
            "generic",
            "style",
            "action",
            "object",
            "scene",
            "pose",
            "outfit",
        ],
        help="Concept type (default: identity).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Target short-edge resolution in px (default: 1024).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum injected token budget (default: 8192).",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Text description of the concept embedded into RefMod metadata.",
    )
    parser.add_argument(
        "--comfy-dir",
        type=Path,
        default=None,
        help="ComfyUI root directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--vae",
        type=Path,
        default=None,
        help="Path to minimax_h3_video_vae_fp16.safetensors (auto-detected if omitted).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory for RefMod file (default: ComfyUI/models/refmods).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Also copy the generated RefMod into the dataset directory (off by default).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories for images, videos, and audio files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan files and display configuration without running extraction.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        generate_refmod(
            folder_input=args.folder,
            name=args.name,
            mode=args.mode,
            concept_type=args.concept_type,
            resolution=args.resolution,
            max_tokens=args.max_tokens,
            description=args.description,
            env_overrides=ComfyUIEnvOverrides(
                comfy_dir=args.comfy_dir,
                vae_path=args.vae,
                refmods_dir=args.output,
            ),
            copy_to_dataset=args.copy,
            recursive=args.recursive,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as err:  # pylint: disable=broad-except
        print(f"\n[ERROR] {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
